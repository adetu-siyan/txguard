import json
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta

# ── CBN thresholds (must match config.py) ────────────────────────────────
NFIU_INDIVIDUAL_THRESHOLD = 5_000_000
ATM_DAILY_MAX = 100_000
WEEKLY_INDIVIDUAL_MAX = 500_000
TIER_1_DAILY_MAX = 30_000
LATE_NIGHT_START = 1
LATE_NIGHT_END = 4

# ── Customer state tracker ────────────────────────────────────────────────
# Maintained in-memory per customer as transactions are processed
# sequentially. Mirrors what a real streaming feature store would hold.
class CustomerState:
    def __init__(self):
        self.transactions = []          # full history, ordered by timestamp
        self.amounts = []               # raw amounts, for mean/std
        self.channels = defaultdict(int)  # channel frequency
        self.hours = defaultdict(int)     # hour-of-day frequency
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
        if not self.amounts:
            return 0
        return sum(self.amounts) / len(self.amounts)

    def std_amount(self):
        if len(self.amounts) < 2:
            return 1  # avoid division by zero
        mean = self.mean_amount()
        variance = sum((a - mean) ** 2 for a in self.amounts) / len(self.amounts)
        return math.sqrt(variance) or 1

    def most_common_channel(self):
        if not self.channels:
            return None
        return max(self.channels, key=self.channels.get)

    def most_common_hour_range(self):
        """Returns the modal 4-hour block this customer usually transacts in."""
        if not self.hours:
            return None
        return max(self.hours, key=self.hours.get)

    def transactions_in_window(self, now, hours):
        cutoff = now - timedelta(hours=hours)
        return [(ts, t) for ts, t in self.transactions if ts >= cutoff]

    def sum_in_window(self, now, hours):
        return sum(t["amount"] for ts, t in self.transactions_in_window(now, hours))

    def count_in_window(self, now, hours):
        return len(self.transactions_in_window(now, hours))

    def cross_account_sum_in_window(self, now, hours):
        """
        Sum of all transactions across ALL accounts belonging to this customer
        within the window. This is the cross-account aggregation that static
        per-account SQL rules structurally cannot compute.
        """
        return self.sum_in_window(now, hours)

    def last_transaction_ts(self):
        if len(self.transactions) < 2:
            return None
        return self.transactions[-2][0]  # second-to-last (last is current)

    def unique_counterparties_in_window(self, now, hours):
        """Approximate: counts unique account_ids transacted from in window."""
        return len({
            t["account_id"]
            for ts, t in self.transactions_in_window(now, hours)
        })

    def account_age_days(self, now):
        if self.first_seen is None:
            return 0
        return (now - self.first_seen).days


# ── Feature Engineering ───────────────────────────────────────────────────
def engineer_features(txn, state, now):
    """
    Takes one raw transaction dict and the current customer state, returns
    a flat feature dict. Called AFTER state.update() so the current
    transaction is already in the history when window features are computed.
    """
    amount = txn["amount"]
    hour = now.hour
    tier = txn["account_tier"]
    channel = txn["channel"]
    txn_type = txn["type"]

    # ── 1. Amount Features ────────────────────────────────────────────────
    amount_log = math.log1p(amount)

    threshold_proximity = amount / NFIU_INDIVIDUAL_THRESHOLD

    # Roundness: 1.0 = perfectly round (e.g. ₦50,000.00), 0.0 = unround
    round_100 = round(amount / 100) * 100
    amount_roundness = 1 - (abs(amount - round_100) / max(amount, 1))

    is_near_threshold = 1 if NFIU_INDIVIDUAL_THRESHOLD * 0.90 <= amount < NFIU_INDIVIDUAL_THRESHOLD else 0
    exceeds_atm_daily = 1 if amount > ATM_DAILY_MAX and channel == "ATM" else 0
    exceeds_tier1_daily = 1 if amount > TIER_1_DAILY_MAX and tier == "TIER_1" else 0

    # ── 2. Temporal Features ──────────────────────────────────────────────
    is_late_night = 1 if LATE_NIGHT_START <= hour <= LATE_NIGHT_END else 0
    is_weekend = 1 if now.weekday() >= 5 else 0

    last_ts = state.last_transaction_ts()
    inter_txn_interval = (now - last_ts).total_seconds() if last_ts else -1

    daily_txn_count = state.count_in_window(now, 24)
    hourly_txn_count = state.count_in_window(now, 1)

    # Velocity: transactions per hour over last 24h window
    daily_velocity = daily_txn_count / 24.0

    # ── 3. Customer Baseline Deviation ────────────────────────────────────
    customer_mean = state.mean_amount()
    customer_std = state.std_amount()

    amount_vs_mean = amount / max(customer_mean, 1)
    amount_zscore = (amount - customer_mean) / customer_std

    usual_channel = state.most_common_channel()
    channel_consistency = 1 if channel == usual_channel else 0

    usual_hour = state.most_common_hour_range()
    hour_consistency = 1 if usual_hour is not None and abs(hour - usual_hour) <= 2 else 0

    # ── 4. Window Aggregations ────────────────────────────────────────────
    sum_1h = state.sum_in_window(now, 1)
    sum_24h = state.sum_in_window(now, 24)
    sum_7d = state.sum_in_window(now, 24 * 7)
    count_1h = state.count_in_window(now, 1)
    count_24h = state.count_in_window(now, 24)

    # Cross-account aggregation — the key structuring signal
    cross_account_sum_24h = state.cross_account_sum_in_window(now, 24)
    cross_account_threshold_ratio = cross_account_sum_24h / NFIU_INDIVIDUAL_THRESHOLD

    # Maximum single transaction in last 24h
    recent_amounts = [t["amount"] for ts, t in state.transactions_in_window(now, 24)]
    max_single_24h = max(recent_amounts) if recent_amounts else 0

    # Coefficient of variation in 24h window (low CoV + high volume = structuring signal)
    if len(recent_amounts) > 1:
        mean_24h = sum(recent_amounts) / len(recent_amounts)
        std_24h = math.sqrt(sum((a - mean_24h) ** 2 for a in recent_amounts) / len(recent_amounts))
        cov_24h = std_24h / max(mean_24h, 1)
    else:
        cov_24h = 0

    # ── 5. Graph / Relational Features ───────────────────────────────────
    accounts_per_customer = len(state.account_ids)
    account_age_days = state.account_age_days(now)
    unique_counterparties_24h = state.unique_counterparties_in_window(now, 24)

    # Tier encoding (ordinal: TIER_1=1, TIER_2=2, TIER_3=3)
    tier_numeric = {"TIER_1": 1, "TIER_2": 2, "TIER_3": 3}.get(tier, 0)

    # Channel encoding
    channel_numeric = {"ATM": 0, "mobile": 1, "POS": 2, "USSD": 3}.get(channel, -1)

    # Transaction type encoding
    type_numeric = {"withdrawal": 0, "transfer": 1, "bill_payment": 2, "airtime": 3}.get(txn_type, -1)

    return {
        # Amount features
        "amount": amount,
        "amount_log": amount_log,
        "threshold_proximity": threshold_proximity,
        "amount_roundness": amount_roundness,
        "is_near_threshold": is_near_threshold,
        "exceeds_atm_daily": exceeds_atm_daily,
        "exceeds_tier1_daily": exceeds_tier1_daily,

        # Temporal features
        "hour_of_day": hour,
        "is_late_night": is_late_night,
        "is_weekend": is_weekend,
        "inter_txn_interval_seconds": inter_txn_interval,
        "daily_txn_count": daily_txn_count,
        "hourly_txn_count": hourly_txn_count,
        "daily_velocity": daily_velocity,

        # Customer baseline deviation
        "amount_vs_customer_mean": amount_vs_mean,
        "amount_zscore": amount_zscore,
        "channel_consistency": channel_consistency,
        "hour_consistency": hour_consistency,

        # Window aggregations
        "sum_1h": sum_1h,
        "sum_24h": sum_24h,
        "sum_7d": sum_7d,
        "count_1h": count_1h,
        "count_24h": count_24h,
        "cross_account_sum_24h": cross_account_sum_24h,
        "cross_account_threshold_ratio": cross_account_threshold_ratio,
        "max_single_24h": max_single_24h,
        "cov_24h": cov_24h,

        # Graph / relational features
        "accounts_per_customer": accounts_per_customer,
        "account_age_days": account_age_days,
        "unique_counterparties_24h": unique_counterparties_24h,
        "tier_numeric": tier_numeric,
        "channel_numeric": channel_numeric,
        "type_numeric": type_numeric,

        # Ground-truth label (for training only — never feed to model at inference)
        "is_suspicious": int(txn.get("is_suspicious", False)),
        "suspicious_typology": txn.get("suspicious_typology") or "normal",
    }


# ── Dataset Builder ───────────────────────────────────────────────────────
def build_feature_dataset(jsonl_path):
    """
    Reads the simulator's .jsonl output, maintains per-customer state,
    and returns a list of feature dicts — one per transaction — ready for
    conversion into a pandas DataFrame and model training.
    """
    customer_states = defaultdict(CustomerState)
    feature_rows = []

    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                txn = json.loads(line)
            except json.JSONDecodeError:
                continue

            customer_id = txn.get("customer_id")
            if not customer_id:
                continue

            try:
                ts_str = txn["timestamp"].replace("Z", "+00:00")
                now = datetime.fromisoformat(ts_str).replace(tzinfo=None)
            except (KeyError, ValueError):
                now = datetime.utcnow()

            state = customer_states[customer_id]

            # Update state FIRST so current txn is included in window features
            state.update(txn, now)

            # Then engineer features against the updated state
            features = engineer_features(txn, state, now)
            features["transaction_id"] = txn.get("transaction_id")
            features["account_id"] = txn.get("account_id")
            features["customer_id"] = customer_id

            feature_rows.append(features)

    return feature_rows


if __name__ == "__main__":
    import csv

    DATA_PATH = os.path.join(
        os.path.dirname(__file__), "..", "data", "simulated_transactions.jsonl"
    )
    OUTPUT_PATH = os.path.join(
        os.path.dirname(__file__), "..", "data", "features.csv"
    )

    print(f"Reading transactions from {DATA_PATH}...")
    rows = build_feature_dataset(DATA_PATH)
    print(f"Engineered features for {len(rows)} transactions.")

    if rows:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved feature matrix to {OUTPUT_PATH}")

        # Quick label distribution check
        suspicious = sum(1 for r in rows if r["is_suspicious"] == 1)
        normal = len(rows) - suspicious
        print(f"\nLabel distribution:")
        print(f"  Normal:     {normal} ({100*normal/len(rows):.1f}%)")
        print(f"  Suspicious: {suspicious} ({100*suspicious/len(rows):.1f}%)")