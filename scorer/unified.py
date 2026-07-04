"""
TxGuard Unified Scoring Pipeline
=================================
Fuses three detection layers into a single risk score per transaction:

  Layer 2 — Isolation Forest  (unsupervised anomaly, PR-AUC 0.306)
  Layer 3 — Random Forest     (supervised transaction-level, PR-AUC 0.865)
  Layer 4 — GraphSAGE GNN     (account graph, cross-account structuring, PR-AUC 0.969)

Fusion weights reflect each model's PR-AUC contribution:
  iso_score * 0.15 + rf_score * 0.40 + gnn_score * 0.45

Output: a structured TxGuardAlert dict with risk score, tier, triggered
rules, regulatory references, and recommended action — ready for Phase 5
agentic investigation layer to consume.

Usage:
    from scorer.unified import TxGuardScorer
    scorer = TxGuardScorer()
    alert = scorer.score(transaction_dict)

Where transaction_dict is a raw transaction in the same format the
simulator produces — see simulator/generators.py _build_txn() for schema.
"""

import os
import json
import pickle
import math
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime, timedelta
from collections import defaultdict
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import add_self_loops, remove_self_loops

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── CBN Thresholds ────────────────────────────────────────────────────────
NFIU_INDIVIDUAL_THRESHOLD = 5_000_000
ATM_DAILY_MAX = 100_000
TIER_1_DAILY_MAX = 30_000
LATE_NIGHT_START = 1
LATE_NIGHT_END = 4

# ── Feature columns (must match features/model.py FEATURE_COLS exactly) ──
FEATURE_COLS = [
    "amount", "amount_log", "threshold_proximity", "amount_roundness",
    "is_near_threshold", "exceeds_atm_daily", "exceeds_tier1_daily",
    "hour_of_day", "is_late_night", "is_weekend",
    "inter_txn_interval_seconds", "daily_txn_count", "hourly_txn_count",
    "daily_velocity", "amount_vs_customer_mean", "amount_zscore",
    "channel_consistency", "hour_consistency",
    "sum_1h", "sum_24h", "sum_7d", "count_1h", "count_24h",
    "cross_account_sum_24h", "cross_account_threshold_ratio",
    "cross_account_sum_6h", "cross_account_ratio_6h",
    "max_single_24h", "cov_24h",
    "accounts_per_customer", "account_age_days",
    "unique_counterparties_24h", "tier_numeric",
    "channel_numeric", "type_numeric",
]

# ── Rule registry (mirrors scorer/main.py RULE_REGISTRY) ─────────────────
RULE_REGISTRY = {
    "NFIU_THRESHOLD_APPROACH": {
        "severity": "HIGH",
        "regulatory_reference": "MLPPA 2022, Section 2 — mandatory NFIU reporting for cash transactions exceeding ₦5,000,000 (individuals) / ₦10,000,000 (corporates)",
        "legal_consequence": "Failure to file mandatory report within 7 days: ₦250,000–₦1,000,000/day fine",
        "recommended_action": "Escalate to MLRO. File STR with NFIU within 7 days.",
    },
    "DAILY_ATM_BREACH": {
        "severity": "HIGH",
        "regulatory_reference": "CBN Circular FPRD/DIR/PUB/CIR/001/011 (effective Jan 1 2026) — ATM cap ₦100,000/day",
        "legal_consequence": "3% excess withdrawal fee on breach amount, split 40% CBN / 60% bank",
        "recommended_action": "Apply excess fee. Notify customer via SMS/USSD.",
    },
    "TRANSFER_VELOCITY": {
        "severity": "HIGH",
        "regulatory_reference": "MLPPA 2022, Section 2(2) — structuring prohibition",
        "legal_consequence": "Criminal offence independent of whether funds are proven illicit",
        "recommended_action": "Freeze outbound transfers. Initiate EDD review.",
    },
    "WEEKLY_WITHDRAWAL_BREACH": {
        "severity": "HIGH",
        "regulatory_reference": "CBN Circular FPRD/DIR/PUB/CIR/001/011 — weekly cap ₦500,000 individual / ₦5,000,000 corporate",
        "legal_consequence": "3%/5% excess withdrawal fee; monthly CBN reporting required",
        "recommended_action": "Apply excess fee. Notify customer.",
    },
    "STRUCTURING_DETECTED": {
        "severity": "HIGH",
        "regulatory_reference": "MLPPA 2022, Section 2(2) — splitting transactions to evade threshold",
        "legal_consequence": "Criminal offence under MLPPA 2022",
        "recommended_action": "Freeze account. File STR with NFIU within 7 days.",
    },
    "BURST_PATTERN": {
        "severity": "MEDIUM",
        "regulatory_reference": "CBN AML/CFT/CPF Regulations 2022 — unusual transaction velocity",
        "legal_consequence": "Potential account takeover. Institutional liability for inadequate monitoring.",
        "recommended_action": "Trigger OTP re-authentication. Alert account owner.",
    },
    "KYC_TIER1_BREACH": {
        "severity": "HIGH",
        "regulatory_reference": "CBN Tiered KYC Framework — Tier 1 daily limit ₦30,000",
        "legal_consequence": "CDD/KYC breach; per-day administrative penalties",
        "recommended_action": "Block transaction. Prompt customer to complete KYC Tier 3+.",
    },
    "ROUND_AMOUNT_PATTERN": {
        "severity": "MEDIUM",
        "regulatory_reference": "FATF/NFIU layering typology — CBN AML/CFT/CPF Regulations 2022",
        "legal_consequence": "Layering red flag; feeds into STR obligation if suspicion substantiated",
        "recommended_action": "Flag for enhanced due diligence.",
    },
    "LATE_NIGHT_HIGH_VALUE": {
        "severity": "MEDIUM",
        "regulatory_reference": "MLPPA 2022, Sections 7-10 — CDD risk profile consistency",
        "legal_consequence": "Unaddressed pattern cited as inadequate ongoing monitoring in CBN examination",
        "recommended_action": "Trigger OTP verification. Log for CDD review.",
    },
}

# ── Fusion weights ────────────────────────────────────────────────────────
# Proportional to each model's PR-AUC. Validated direction: GNN highest
# because it catches the cross-account structuring case others miss.
ISO_WEIGHT = 0.15
RF_WEIGHT  = 0.40
GNN_WEIGHT = 0.45

# ── Risk tiers ────────────────────────────────────────────────────────────
def risk_tier(score):
    if score >= 70:
        return "HIGH"
    elif score >= 40:
        return "MEDIUM"
    return "LOW"


# ── Customer state (mirrors features/engineer.py CustomerState) ───────────
class CustomerState:
    def __init__(self):
        self.transactions = []
        self.amounts = []
        self.channels = defaultdict(int)
        self.hours = defaultdict(int)
        self.first_seen = None
        self.account_ids = set()

    def update(self, txn, ts):
        self.transactions.append((ts, txn))
        self.amounts.append(txn["amount"])
        self.channels[txn["channel"]] += 1
        self.hours[ts.hour] += 1
        self.account_ids.add(txn["account_id"])
        if self.first_seen is None:
            self.first_seen = ts

    def mean_amount(self):
        return sum(self.amounts) / len(self.amounts) if self.amounts else 0

    def std_amount(self):
        if len(self.amounts) < 2:
            return 1
        mean = self.mean_amount()
        return math.sqrt(sum((a - mean) ** 2 for a in self.amounts) / len(self.amounts)) or 1

    def most_common_channel(self):
        return max(self.channels, key=self.channels.get) if self.channels else None

    def most_common_hour(self):
        return max(self.hours, key=self.hours.get) if self.hours else None

    def txns_in_window(self, now, hours):
        cutoff = now - timedelta(hours=hours)
        return [(ts, t) for ts, t in self.transactions if ts >= cutoff]

    def sum_in_window(self, now, hours):
        return sum(t["amount"] for _, t in self.txns_in_window(now, hours))

    def count_in_window(self, now, hours):
        return len(self.txns_in_window(now, hours))

    def last_ts(self):
        return self.transactions[-2][0] if len(self.transactions) >= 2 else None

    def account_age_days(self, now):
        return (now - self.first_seen).days if self.first_seen else 0


# ── Feature engineering (single transaction) ─────────────────────────────
def engineer_single(txn, state, now):
    amount = txn["amount"]
    hour = now.hour
    tier = txn.get("account_tier", "TIER_2")
    channel = txn.get("channel", "mobile")
    txn_type = txn.get("type", "transfer")

    amount_log = math.log1p(amount)
    threshold_proximity = amount / NFIU_INDIVIDUAL_THRESHOLD
    round_100 = round(amount / 100) * 100
    amount_roundness = 1 - (abs(amount - round_100) / max(amount, 1))
    is_near_threshold = int(NFIU_INDIVIDUAL_THRESHOLD * 0.90 <= amount < NFIU_INDIVIDUAL_THRESHOLD)
    exceeds_atm_daily = int(amount > ATM_DAILY_MAX and channel == "ATM")
    exceeds_tier1_daily = int(amount > TIER_1_DAILY_MAX and tier == "TIER_1")

    is_late_night = int(LATE_NIGHT_START <= hour <= LATE_NIGHT_END)
    is_weekend = int(now.weekday() >= 5)
    last = state.last_ts()
    inter_txn_interval = (now - last).total_seconds() if last else -1
    daily_txn_count = state.count_in_window(now, 24)
    hourly_txn_count = state.count_in_window(now, 1)
    daily_velocity = daily_txn_count / 24.0

    customer_mean = state.mean_amount()
    customer_std = state.std_amount()
    amount_vs_mean = amount / max(customer_mean, 1)
    amount_zscore = (amount - customer_mean) / customer_std

    usual_channel = state.most_common_channel()
    channel_consistency = int(channel == usual_channel)
    usual_hour = state.most_common_hour()
    hour_consistency = int(usual_hour is not None and abs(hour - usual_hour) <= 2)

    sum_1h = state.sum_in_window(now, 1)
    sum_24h = state.sum_in_window(now, 24)
    sum_7d = state.sum_in_window(now, 24 * 7)
    count_1h = state.count_in_window(now, 1)
    count_24h = state.count_in_window(now, 24)
    cross_sum_24h = sum_24h
    cross_ratio_24h = cross_sum_24h / NFIU_INDIVIDUAL_THRESHOLD
    cross_sum_6h = state.sum_in_window(now, 6)
    cross_ratio_6h = cross_sum_6h / NFIU_INDIVIDUAL_THRESHOLD

    recent = [t["amount"] for _, t in state.txns_in_window(now, 24)]
    max_single_24h = max(recent) if recent else 0
    if len(recent) > 1:
        m = sum(recent) / len(recent)
        s = math.sqrt(sum((a - m) ** 2 for a in recent) / len(recent))
        cov_24h = s / max(m, 1)
    else:
        cov_24h = 0

    accounts_per_customer = len(state.account_ids)
    account_age_days = state.account_age_days(now)
    unique_counterparties_24h = len({
        t["account_id"] for _, t in state.txns_in_window(now, 24)
    })

    tier_numeric = {"TIER_1": 1, "TIER_2": 2, "TIER_3": 3}.get(tier, 0)
    channel_numeric = {"ATM": 0, "mobile": 1, "POS": 2, "USSD": 3}.get(channel, -1)
    type_numeric = {"withdrawal": 0, "transfer": 1, "bill_payment": 2, "airtime": 3}.get(txn_type, -1)

    return [
        amount, amount_log, threshold_proximity, amount_roundness,
        is_near_threshold, exceeds_atm_daily, exceeds_tier1_daily,
        hour, is_late_night, is_weekend,
        inter_txn_interval, daily_txn_count, hourly_txn_count, daily_velocity,
        amount_vs_mean, amount_zscore, channel_consistency, hour_consistency,
        sum_1h, sum_24h, sum_7d, count_1h, count_24h,
        cross_sum_24h, cross_ratio_24h, cross_sum_6h, cross_ratio_6h,
        max_single_24h, cov_24h,
        accounts_per_customer, account_age_days, unique_counterparties_24h,
        tier_numeric, channel_numeric, type_numeric,
    ]


# ── GraphSAGE model class (mirrors features/gnn.py) ──────────────────────
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels // 2)
        self.classifier = Linear(hidden_channels // 2, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


# ── Rule checker (deterministic CBN rules) ───────────────────────────────
def check_rules(txn, state, now):
    """
    Runs the 9 deterministic CBN rules against the current transaction
    and customer state. Returns list of triggered rule names.
    These mirror the Stream Analytics SQL rules but run in Python so
    the unified scorer can include them without Azure dependency.
    """
    triggered = []
    amount = txn["amount"]
    channel = txn.get("channel", "mobile")
    tier = txn.get("account_tier", "TIER_2")
    txn_type = txn.get("type", "transfer")
    hour = now.hour

    # Rule 1: NFIU threshold approach
    cross_sum_24h = state.sum_in_window(now, 24)
    if cross_sum_24h > NFIU_INDIVIDUAL_THRESHOLD * 0.90:
        triggered.append("NFIU_THRESHOLD_APPROACH")

    # Rule 2: ATM daily breach
    if channel == "ATM" and txn_type == "withdrawal" and amount > ATM_DAILY_MAX:
        triggered.append("DAILY_ATM_BREACH")

    # Rule 3: Burst pattern (5+ transactions in last 60 seconds)
    recent_60s = [t for ts, t in state.txns_in_window(now, 1/60) ]
    if len(recent_60s) >= 5:
        triggered.append("BURST_PATTERN")

    # Rule 4: Structuring — multiple transactions summing near threshold
    count_24h = state.count_in_window(now, 24)
    if (count_24h >= 3 and
        cross_sum_24h > NFIU_INDIVIDUAL_THRESHOLD * 0.85 and
        amount < NFIU_INDIVIDUAL_THRESHOLD):
        triggered.append("STRUCTURING_DETECTED")

    # Rule 5: Weekly withdrawal breach (approximate — 7-day window)
    sum_7d = state.sum_in_window(now, 24 * 7)
    if txn_type == "withdrawal" and sum_7d > 500_000:
        triggered.append("WEEKLY_WITHDRAWAL_BREACH")

    # Rule 6: Late night high-value withdrawal
    if (LATE_NIGHT_START <= hour <= LATE_NIGHT_END and
        txn_type == "withdrawal" and
        amount > 50_000):
        triggered.append("LATE_NIGHT_HIGH_VALUE")

    # Rule 7: Transfer velocity
    count_1h = state.count_in_window(now, 1)
    if txn_type == "transfer" and count_1h >= 5:
        triggered.append("TRANSFER_VELOCITY")

    # Rule 8: KYC Tier 1 breach
    if tier == "TIER_1" and amount > TIER_1_DAILY_MAX:
        triggered.append("KYC_TIER1_BREACH")

    # Rule 9: Round amount pattern
    recent_24h = state.txns_in_window(now, 24)
    round_count = sum(
        1 for _, t in recent_24h
        if txn_type == "transfer" and int(t["amount"]) % 10000 == 0
    )
    if round_count >= 3:
        triggered.append("ROUND_AMOUNT_PATTERN")

    return triggered


# ── Main scorer class ─────────────────────────────────────────────────────
class TxGuardScorer:
    """
    Unified TxGuard scoring pipeline. Loads all three trained models
    once at initialization, maintains per-customer state in memory,
    and scores each incoming transaction in real time.

    Usage:
        scorer = TxGuardScorer()
        alert = scorer.score(txn_dict)
    """

    def __init__(self):
        print("Loading TxGuard models...")

        # Isolation Forest
        with open(os.path.join(MODEL_DIR, "isolation_forest.pkl"), "rb") as f:
            self.iso = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
            self.rf_scaler = pickle.load(f)

        # Random Forest
        with open(os.path.join(MODEL_DIR, "random_forest.pkl"), "rb") as f:
            self.rf = pickle.load(f)

        # GNN
        with open(os.path.join(MODEL_DIR, "gnn_feature_count.json"), "r") as f:
            gnn_meta = json.load(f)
        self.gnn = GraphSAGE(
            in_channels=gnn_meta["in_channels"],
            hidden_channels=64,
            out_channels=2,
            dropout=0.3
        )
        self.gnn.load_state_dict(
            torch.load(os.path.join(MODEL_DIR, "graphsage.pt"), map_location="cpu")
        )
        self.gnn.eval()

        with open(os.path.join(MODEL_DIR, "gnn_scaler.pkl"), "rb") as f:
            self.gnn_scaler = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "gnn_accounts.json"), "r") as f:
            self.known_accounts = json.load(f)

        # In-memory customer state (grows as transactions arrive)
        self.customer_states = defaultdict(CustomerState)

        # Account graph state (updated incrementally)
        self.account_features = {}   # account_id -> feature vector (account-level)
        self.account_labels = {}     # account_id -> current risk label
        self.customer_accounts = defaultdict(set)  # customer_id -> set of account_ids

        print("All models loaded. TxGuardScorer ready.")

    def _get_timestamp(self, txn):
        try:
            ts_str = txn["timestamp"].replace("Z", "+00:00")
            return datetime.fromisoformat(ts_str).replace(tzinfo=None)
        except (KeyError, ValueError):
            return datetime.utcnow()

    def _iso_score(self, features_scaled):
        raw = -self.iso.score_samples([features_scaled])[0]
        # Normalize to 0-1 using the model's contamination threshold
        return float(np.clip(raw / 1.5, 0, 1))

    def _rf_score(self, features_scaled):
        proba = self.rf.predict_proba([features_scaled])[0]
        return float(proba[1])

    def _gnn_score(self, account_id, customer_id):
        """
        Runs GNN inference on the current account graph state.
        Returns the suspicion probability for the target account node.
        """
        all_accounts = list(self.account_features.keys())
        if account_id not in all_accounts:
            return 0.5  # unknown account — return neutral score

        account_to_idx = {acc: i for i, acc in enumerate(all_accounts)}
        num_nodes = len(all_accounts)

        # Build node feature matrix
        feat_matrix = np.array([
            self.account_features[acc] for acc in all_accounts
        ], dtype=np.float32)
        feat_matrix = np.nan_to_num(feat_matrix, nan=0.0)

        try:
            feat_matrix = self.gnn_scaler.transform(feat_matrix)
        except Exception:
            pass

        feat_matrix = np.nan_to_num(feat_matrix, nan=0.0)

        # Build edges (shared customer identity)
        edge_sources, edge_targets = [], []
        for cust, accs in self.customer_accounts.items():
            accs_list = [a for a in accs if a in account_to_idx]
            for i in range(len(accs_list)):
                for j in range(len(accs_list)):
                    if i != j:
                        edge_sources.append(account_to_idx[accs_list[i]])
                        edge_targets.append(account_to_idx[accs_list[j]])

        if edge_sources:
            edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        x = torch.tensor(feat_matrix, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)

        self.gnn.eval()
        with torch.no_grad():
            out = self.gnn(data.x, data.edge_index)
            proba = F.softmax(out, dim=1)[:, 1].numpy()
            proba = np.nan_to_num(proba, nan=0.5)

        target_idx = account_to_idx[account_id]
        return float(proba[target_idx])

    def _update_account_graph(self, account_id, customer_id, state, now):
        """
        Updates the account-level feature vector for this account.
        Called after CustomerState is updated so features reflect
        the latest transaction.
        """
        self.customer_accounts[customer_id].add(account_id)
        amounts = state.amounts
        if not amounts:
            return

        mean_amount = np.mean(amounts)
        std_amount = np.std(amounts) if len(amounts) > 1 else 0
        max_amount = np.max(amounts)

        # Build 33-dim account feature vector (matches gnn.py build_account_features)
        cross_sum_24h = state.sum_in_window(now, 24)
        cross_sum_6h = state.sum_in_window(now, 6)
        feat = [
            math.log1p(mean_amount),
            math.log1p(std_amount),
            math.log1p(max_amount),
            math.log1p(min(amounts)),
            std_amount / max(mean_amount, 1),
            cross_sum_24h / NFIU_INDIVIDUAL_THRESHOLD,
            state.sum_in_window(now, 24) / NFIU_INDIVIDUAL_THRESHOLD,
            math.log1p(cross_sum_24h),
            cross_sum_24h / NFIU_INDIVIDUAL_THRESHOLD,
            math.log1p(cross_sum_6h),
            cross_sum_6h / NFIU_INDIVIDUAL_THRESHOLD,
            (max_amount - mean_amount) / max(std_amount, 1),
            max_amount / max(mean_amount, 1),
            state.count_in_window(now, 24),
            state.count_in_window(now, 1),
            state.count_in_window(now, 24) / 24.0,
            0.0,  # late_night_rate (simplified)
            0.0,  # weekend_rate (simplified)
            now.hour / 24.0,
            0.0,  # any_atm_breach
            0.0,  # any_tier1_breach
            0.0,  # any_near_threshold
            0.0,  # breach_rate
            len(state.account_ids),
            len(state.account_ids),
            math.log1p(len(amounts)),
            0.5,  # mean_roundness (simplified)
            0.5,  # max_roundness (simplified)
            0.5,  # channel_consistency (simplified)
            1.5,  # channel_numeric_mean (simplified)
            1.5,  # type_numeric_mean (simplified)
            math.log1p(state.sum_in_window(now, 1)),
            math.log1p(state.sum_in_window(now, 24)),
        ]

        self.account_features[account_id] = [
            0.0 if (math.isnan(v) or math.isinf(v)) else v for v in feat
        ]

    def score(self, txn):
        """
        Scores one transaction and returns a TxGuardAlert dict.

        Args:
            txn: dict with keys matching simulator output schema:
                 account_id, customer_id, account_tier, amount,
                 type, channel, timestamp, etc.

        Returns:
            dict: TxGuardAlert with risk score, tier, triggered rules,
                  regulatory references, and recommended action.
        """
        account_id = txn.get("account_id", "UNKNOWN")
        customer_id = txn.get("customer_id", "UNKNOWN")
        now = self._get_timestamp(txn)

        # Update customer state
        state = self.customer_states[customer_id]
        state.update(txn, now)

        # Update account graph
        self._update_account_graph(account_id, customer_id, state, now)

        # ── Feature engineering ───────────────────────────────────────────
        features = engineer_single(txn, state, now)
        features_array = np.array(features).reshape(1, -1)
        features_scaled = self.rf_scaler.transform(features_array)[0]

        # ── Model scores ──────────────────────────────────────────────────
        iso_score = self._iso_score(features_scaled)
        rf_score = self._rf_score(features_scaled)
        gnn_score = self._gnn_score(account_id, customer_id)

        # ── Fusion ────────────────────────────────────────────────────────
        fused = (
            ISO_WEIGHT * iso_score +
            RF_WEIGHT  * rf_score  +
            GNN_WEIGHT * gnn_score
        )
        txguard_risk_score = round(fused * 100, 1)
        tier = risk_tier(txguard_risk_score)

        # ── Rule checks ───────────────────────────────────────────────────
        triggered_rules = check_rules(txn, state, now)

        # ── Regulatory synthesis ──────────────────────────────────────────
        if triggered_rules:
            # Use the highest-severity rule's references
            highest = triggered_rules[0]
            reg_info = RULE_REGISTRY.get(highest, {})
            regulatory_reference = reg_info.get("regulatory_reference", "N/A")
            legal_consequence = reg_info.get("legal_consequence", "N/A")
            recommended_action = reg_info.get("recommended_action", "Review and escalate if warranted.")
        else:
            regulatory_reference = "No hard rule triggered — ML anomaly detection flag"
            legal_consequence = "Review warranted based on behavioral pattern"
            recommended_action = "Enhanced monitoring. Escalate to MLRO if pattern persists."

        # ── Alert output ──────────────────────────────────────────────────
        alert = {
            "transaction_id": txn.get("transaction_id", "UNKNOWN"),
            "account_id": account_id,
            "customer_id": customer_id,
            "account_tier": txn.get("account_tier", "UNKNOWN"),
            "amount": txn.get("amount", 0),
            "timestamp": txn.get("timestamp", now.isoformat()),
            # Model scores
            "iso_score": round(iso_score, 4),
            "rf_score": round(rf_score, 4),
            "gnn_score": round(gnn_score, 4),
            # Fused output
            "txguard_risk_score": txguard_risk_score,
            "risk_tier": tier,
            # Rule engine
            "triggered_rules": triggered_rules,
            # Regulatory
            "regulatory_reference": regulatory_reference,
            "legal_consequence": legal_consequence,
            "recommended_action": recommended_action,
            # Metadata
            "scored_at": datetime.utcnow().isoformat() + "Z",
        }

        return alert


# ── Demo runner ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json as _json

    scorer = TxGuardScorer()

    # Load simulator output and score each transaction
    data_path = os.path.join(BASE_DIR, "data", "simulated_transactions.jsonl")

    if not os.path.exists(data_path):
        print(f"No data found at {data_path}. Run simulator first.")
    else:
        print(f"\nScoring transactions from {data_path}...")
        print("=" * 70)

        high_alerts = []
        total = 0
        flagged = 0

        with open(data_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    txn = _json.loads(line)
                except _json.JSONDecodeError:
                    continue

                alert = scorer.score(txn)
                total += 1

                if alert["risk_tier"] in ("HIGH", "MEDIUM"):
                    flagged += 1
                    high_alerts.append(alert)

                    if alert["risk_tier"] == "HIGH":
                        print(f"\n  [HIGH ALERT] Score: {alert['txguard_risk_score']}/100")
                        print(f"  Account:    {alert['account_id']} ({alert['customer_id']})")
                        print(f"  Amount:     ₦{alert['amount']:,.2f}")
                        print(f"  ISO:        {alert['iso_score']:.3f} | "
                              f"RF: {alert['rf_score']:.3f} | "
                              f"GNN: {alert['gnn_score']:.3f}")
                        print(f"  Rules:      {', '.join(alert['triggered_rules']) or 'None (ML flag)'}")
                        print(f"  Action:     {alert['recommended_action']}")
                        print("  " + "-" * 66)

        print(f"\n{'='*70}")
        print(f"  Total transactions scored: {total}")
        print(f"  Flagged (HIGH/MEDIUM):     {flagged} ({100*flagged/max(total,1):.1f}%)")
        print(f"  HIGH alerts:               {sum(1 for a in high_alerts if a['risk_tier']=='HIGH')}")
        print(f"  MEDIUM alerts:             {sum(1 for a in high_alerts if a['risk_tier']=='MEDIUM')}")
        print(f"{'='*70}")
