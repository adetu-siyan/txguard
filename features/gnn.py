import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import add_self_loops, remove_self_loops
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, average_precision_score,
)
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FEATURES_PATH = os.path.join(DATA_DIR, "features.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


# ── Account-level feature builder ─────────────────────────────────────────
# Instead of averaging per-transaction rolling window features (which
# produces inflated, meaningless node vectors), we compute account-level
# summary statistics that actually describe an account's overall behavior.
# This is the correct representation for GNN node features.

def build_account_features(df):
    """
    Computes one feature vector per account from its transaction history.
    Returns: (node_features array, node_labels array, account_list)
    """
    accounts = df["account_id"].unique().tolist()
    account_to_idx = {acc: i for i, acc in enumerate(accounts)}
    num_nodes = len(accounts)

    # Per-account aggregations
    records = []
    for acc in accounts:
        acc_df = df[df["account_id"] == acc]

        amounts = acc_df["amount"].values
        n = len(amounts)

        # Amount statistics
        mean_amount = amounts.mean()
        std_amount = amounts.std() if n > 1 else 0.0
        max_amount = amounts.max()
        min_amount = amounts.min()
        # Coefficient of variation — low CoV with high volume = structuring signal
        cov_amount = std_amount / max(mean_amount, 1)

        # Threshold proximity — how close did this account get to ₦5M?
        max_threshold_proximity = acc_df["threshold_proximity"].max()
        mean_threshold_proximity = acc_df["threshold_proximity"].mean()

        # Cross-account signals — peak values (not average)
        max_cross_sum_24h = acc_df["cross_account_sum_24h"].max()
        max_cross_ratio_24h = acc_df["cross_account_threshold_ratio"].max()
        max_cross_sum_6h = acc_df["cross_account_sum_6h"].max()
        max_cross_ratio_6h = acc_df["cross_account_ratio_6h"].max()

        # Anomaly signals
        max_zscore = acc_df["amount_zscore"].max()
        max_vs_mean = acc_df["amount_vs_customer_mean"].max()

        # Velocity signals
        max_daily_count = acc_df["daily_txn_count"].max()
        max_hourly_count = acc_df["hourly_txn_count"].max()
        max_daily_velocity = acc_df["daily_velocity"].max()

        # Temporal signals
        late_night_rate = acc_df["is_late_night"].mean()
        weekend_rate = acc_df["is_weekend"].mean()
        mean_hour = acc_df["hour_of_day"].mean()

        # Regulatory breach flags
        any_atm_breach = int(acc_df["exceeds_atm_daily"].any())
        any_tier1_breach = int(acc_df["exceeds_tier1_daily"].any())
        any_near_threshold = int(acc_df["is_near_threshold"].any())
        breach_rate = acc_df["exceeds_atm_daily"].mean() + acc_df["exceeds_tier1_daily"].mean()

        # Account properties
        tier_numeric = acc_df["tier_numeric"].iloc[0]
        accounts_per_customer = acc_df["accounts_per_customer"].iloc[0]
        txn_count = n
        log_txn_count = np.log1p(n)

        # Roundness — structuring often uses round amounts
        mean_roundness = acc_df["amount_roundness"].mean()
        max_roundness = acc_df["amount_roundness"].max()

        # Channel and type diversity (low diversity = rigid pattern = suspicious)
        channel_consistency = acc_df["channel_consistency"].mean()
        channel_numeric_mean = acc_df["channel_numeric"].mean()
        type_numeric_mean = acc_df["type_numeric"].mean()

        # Window sum peaks — not averages, but peaks (max suspicious exposure)
        max_sum_1h = acc_df["sum_1h"].max()
        max_sum_24h = acc_df["sum_24h"].max()

        feature_vector = [
            # Amount stats
            np.log1p(mean_amount),
            np.log1p(std_amount),
            np.log1p(max_amount),
            np.log1p(min_amount),
            cov_amount,
            # Threshold proximity
            max_threshold_proximity,
            mean_threshold_proximity,
            # Cross-account (peak, not average)
            np.log1p(max_cross_sum_24h),
            max_cross_ratio_24h,
            np.log1p(max_cross_sum_6h),
            max_cross_ratio_6h,
            # Anomaly signals
            max_zscore,
            max_vs_mean,
            # Velocity
            max_daily_count,
            max_hourly_count,
            max_daily_velocity,
            # Temporal
            late_night_rate,
            weekend_rate,
            mean_hour / 24.0,  # normalize to 0-1
            # Regulatory breaches
            any_atm_breach,
            any_tier1_breach,
            any_near_threshold,
            breach_rate,
            # Account properties
            tier_numeric,
            accounts_per_customer,
            log_txn_count,
            # Roundness
            mean_roundness,
            max_roundness,
            # Channel/type
            channel_consistency,
            channel_numeric_mean,
            type_numeric_mean,
            # Window peaks (log-scaled to tame large values)
            np.log1p(max_sum_1h),
            np.log1p(max_sum_24h),
        ]

        # Replace any NaN/Inf
        feature_vector = [
            0.0 if (np.isnan(v) or np.isinf(v)) else float(v)
            for v in feature_vector
        ]

        # Label: suspicious if ANY transaction was suspicious
        label = int(acc_df["is_suspicious"].any())

        records.append((acc, feature_vector, label))

    node_features = np.array([r[1] for r in records], dtype=np.float32)
    node_labels = np.array([r[2] for r in records], dtype=np.int64)

    return node_features, node_labels, accounts, account_to_idx


def build_graph(df, scaler=None, fit_scaler=False):
    node_features, node_labels, accounts, account_to_idx = build_account_features(df)
    num_nodes = len(accounts)

    # Normalize
    if fit_scaler:
        scaler = StandardScaler()
        node_features = scaler.fit_transform(node_features)
    elif scaler is not None:
        node_features = scaler.transform(node_features)

    node_features = np.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Edges ─────────────────────────────────────────────────────────────
    edge_sources = []
    edge_targets = []

    # Primary: shared customer identity (the structural fraud signal)
    customer_to_accounts = defaultdict(list)
    for _, row in df.drop_duplicates("account_id").iterrows():
        cust = row.get("customer_id")
        acc = row["account_id"]
        if cust and acc in account_to_idx:
            customer_to_accounts[str(cust)].append(acc)

    for cust, accs in customer_to_accounts.items():
        if len(accs) > 1:
            for i in range(len(accs)):
                for j in range(len(accs)):
                    if i != j:
                        edge_sources.append(account_to_idx[accs[i]])
                        edge_targets.append(account_to_idx[accs[j]])

    # Secondary: accounts with similar tier and high cross-account ratios
    # (catches coordinated structuring between unlinked accounts)
    suspicious_accounts = [
        accounts[i] for i in range(num_nodes)
        if node_labels[i] == 1
    ]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # Connect accounts in the same tier that both show
            # high cross-account ratios — potential unlinked coordination
            tier_i = df[df["account_id"] == accounts[i]]["tier_numeric"].iloc[0]
            tier_j = df[df["account_id"] == accounts[j]]["tier_numeric"].iloc[0]
            ratio_i = df[df["account_id"] == accounts[i]]["cross_account_ratio_6h"].max()
            ratio_j = df[df["account_id"] == accounts[j]]["cross_account_ratio_6h"].max()
            if tier_i == tier_j and ratio_i > 0.5 and ratio_j > 0.5:
                edge_sources.extend([i, j])
                edge_targets.extend([j, i])

    # Deduplicate
    if edge_sources:
        edge_pairs = list(set(zip(edge_sources, edge_targets)))
        srcs, tgts = zip(*edge_pairs)
        edge_index = torch.tensor([list(srcs), list(tgts)], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Self-loops — ensures every node aggregates its own features
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(node_labels, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)

    print(f"\nGraph built:")
    print(f"  Nodes (accounts):    {data.num_nodes}")
    print(f"  Edges:               {data.num_edges}")
    print(f"  Node features:       {data.num_node_features}")
    print(f"  Suspicious nodes:    {int(y.sum())}")
    print(f"  Normal nodes:        {int((y == 0).sum())}")
    print(f"  Max feature value:   {np.abs(node_features).max():.4f}")
    print(f"  NaN in features:     {np.isnan(node_features).any()}")

    return data, accounts, scaler


# ── GraphSAGE ─────────────────────────────────────────────────────────────
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
        x = self.classifier(x)
        return x


def train_epoch(model, data, optimizer, train_mask, class_weights):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    if torch.isnan(out).any():
        return float("nan")
    loss = F.cross_entropy(
        out[train_mask], data.y[train_mask], weight=class_weights
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return float(loss)


def evaluate_gnn(model, data, mask, mask_name):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        proba = F.softmax(out, dim=1)[:, 1].numpy()
    proba = np.nan_to_num(proba, nan=0.0)
    preds = (proba >= 0.3).astype(int)
    y_true = data.y[mask].numpy()
    y_pred = preds[mask]
    y_scores = proba[mask]

    if y_true.sum() == 0:
        print(f"\n  {mask_name}: No suspicious nodes in this split.")
        return {}

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_scores)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    print(f"\n{'='*55}")
    print(f"  GraphSAGE — {mask_name}")
    print(f"{'='*55}")
    print(f"  Precision:  {precision:.3f}  ({tp} TP, {fp} FP)")
    print(f"  Recall:     {recall:.3f}  ({fn} missed)")
    print(f"  F1 Score:   {f1:.3f}")
    print(f"  Accuracy:   {accuracy:.3f}")
    print(f"  PR-AUC:     {pr_auc:.3f}")
    print(f"  TP={tp} | FP={fp} | FN={fn} | TN={tn}")
    return {"precision": precision, "recall": recall, "f1": f1, "pr_auc": pr_auc}


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("Loading feature matrix...")
    df = pd.read_csv(FEATURES_PATH)
    print(f"Loaded {len(df)} transactions | "
          f"{int(df['is_suspicious'].sum())} suspicious | "
          f"{int((df['is_suspicious']==0).sum())} normal")

    print("\nBuilding transaction graph...")
    data, accounts, scaler = build_graph(df, fit_scaler=True)

    num_nodes = data.num_nodes
    split_idx = int(num_nodes * 0.75)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:split_idx] = True
    test_mask[split_idx:] = True

    train_suspicious = int(data.y[train_mask].sum())
    test_suspicious = int(data.y[test_mask].sum())
    print(f"\nTrain nodes: {int(train_mask.sum())} | suspicious: {train_suspicious}")
    print(f"Test nodes:  {int(test_mask.sum())} | suspicious: {test_suspicious}")

    if train_suspicious == 0:
        print("ERROR: No suspicious nodes in training set.")
        return

    n_normal = int((data.y[train_mask] == 0).sum())
    weight_suspicious = min(n_normal / train_suspicious, 10.0)
    class_weights = torch.tensor([1.0, weight_suspicious], dtype=torch.float)
    print(f"Class weights: normal=1.0, suspicious={weight_suspicious:.1f}")

    in_channels = data.num_node_features
    model = GraphSAGE(
        in_channels=in_channels,
        hidden_channels=64,
        out_channels=2,
        dropout=0.3
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: GraphSAGE | {in_channels}→64→32→2 | {total_params:,} params")
    print(f"\nTraining for up to 150 epochs...")

    best_f1 = 0
    best_state = None
    no_improve = 0

    for epoch in range(1, 151):
        loss = train_epoch(model, data, optimizer, train_mask, class_weights)
        scheduler.step()

        if np.isnan(loss):
            print(f"  Epoch {epoch}: NaN loss — stopping.")
            break

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                proba = F.softmax(out, dim=1)[:, 1].numpy()
                proba = np.nan_to_num(proba, nan=0.0)
                preds = (proba >= 0.3).astype(int)
                y_train = data.y[train_mask].numpy()
                train_f1 = f1_score(y_train, preds[train_mask], zero_division=0)

            print(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | Train F1: {train_f1:.3f}")

            if train_f1 > best_f1:
                best_f1 = train_f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 4:
                    print(f"  Early stopping at epoch {epoch}")
                    break

    if best_state:
        model.load_state_dict(best_state)

    print("\nEvaluating on training set:")
    train_metrics = evaluate_gnn(model, data, train_mask, "Train")

    print("\nEvaluating on test set:")
    test_metrics = evaluate_gnn(model, data, test_mask, "Test")

    print(f"\n{'='*55}")
    print(f"  Model Comparison Summary")
    print(f"{'='*55}")
    print(f"  {'Model':<35} {'PR-AUC':>8} {'F1':>8} {'Recall':>8}")
    print(f"  {'-'*55}")
    print(f"  {'Isolation Forest':<35} {'0.306':>8} {'0.293':>8} {'0.362':>8}")
    print(f"  {'Random Forest + SMOTE':<35} {'0.865':>8} {'0.748':>8} {'0.851':>8}")
    if test_metrics:
        print(f"  {'GraphSAGE (GNN)':<35} "
              f"{test_metrics.get('pr_auc', 0):>8.3f} "
              f"{test_metrics.get('f1', 0):>8.3f} "
              f"{test_metrics.get('recall', 0):>8.3f}")
    print(f"{'='*55}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "graphsage.pt"))
    with open(os.path.join(MODEL_DIR, "gnn_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, "gnn_accounts.json"), "w") as f:
        json.dump(accounts, f)
    with open(os.path.join(MODEL_DIR, "gnn_feature_count.json"), "w") as f:
        json.dump({"in_channels": in_channels}, f)

    print(f"\nModel saved to {MODEL_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()