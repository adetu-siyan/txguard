import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, average_precision_score,
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FEATURES_PATH = os.path.join(DATA_DIR, "features.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

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


def evaluate(y_true, y_pred, y_scores, model_name, threshold=None):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_scores)

    tp = sum(1 for a, p in zip(y_true, y_pred) if a == 1 and p == 1)
    fp = sum(1 for a, p in zip(y_true, y_pred) if a == 0 and p == 1)
    fn = sum(1 for a, p in zip(y_true, y_pred) if a == 1 and p == 0)
    tn = sum(1 for a, p in zip(y_true, y_pred) if a == 0 and p == 0)

    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    if threshold:
        print(f"  Threshold:  {threshold}")
    print(f"  Precision:  {precision:.3f}  ({tp} true positives, {fp} false positives)")
    print(f"  Recall:     {recall:.3f}  ({fn} missed suspicious events)")
    print(f"  F1 Score:   {f1:.3f}")
    print(f"  Accuracy:   {accuracy:.3f}")
    print(f"  PR-AUC:     {pr_auc:.3f}")
    print(f"  TP={tp} | FP={fp} | FN={fn} | TN={tn}")


def main():
    print("Loading feature matrix...")
    df = pd.read_csv(FEATURES_PATH)
    print(
        f"Loaded {len(df)} transactions | "
        f"{df['is_suspicious'].sum()} suspicious | "
        f"{(df['is_suspicious']==0).sum()} normal"
    )

    X = df[FEATURE_COLS].values
    y = df["is_suspicious"].values

    # Temporal split — no shuffle
    split_idx = int(len(X) * 0.75)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
    print(f"Train suspicious: {y_train.sum()} | Test suspicious: {y_test.sum()}")

    # Normalize — fit on train only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── Isolation Forest (unsupervised) ───────────────────────────────────
    print("\nTraining Isolation Forest on normal transactions only...")
    contamination = float(y_train.sum()) / len(y_train)
    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    X_normal_train = X_train_scaled[y_train == 0]
    iso.fit(X_normal_train)

    iso_scores_test = -iso.score_samples(X_test_scaled)
    iso_threshold = np.percentile(iso_scores_test, (1 - contamination) * 100)
    iso_preds = (iso_scores_test >= iso_threshold).astype(int)
    evaluate(y_test, iso_preds, iso_scores_test,
             "Isolation Forest (Unsupervised)", threshold=f"{iso_threshold:.4f}")

    # ── Random Forest (supervised + SMOTE) ───────────────────────────────
    print("\nApplying SMOTE to training set...")
    sm = SMOTE(random_state=42, k_neighbors=min(5, int(y_train.sum()) - 1))
    X_train_smote, y_train_smote = sm.fit_resample(X_train_scaled, y_train)
    print(
        f"After SMOTE: {len(X_train_smote)} train samples | "
        f"{y_train_smote.sum()} suspicious"
    )

    print("\nTraining Random Forest classifier...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_smote, y_train_smote)

    rf_scores_test = rf.predict_proba(X_test_scaled)[:, 1]
    rf_preds = (rf_scores_test >= 0.3).astype(int)
    evaluate(y_test, rf_preds, rf_scores_test,
             "Random Forest (Supervised + SMOTE)", threshold=0.3)

    # ── Full feature importance ───────────────────────────────────────────
    print("\n  Full feature importance (Random Forest):")
    importances = rf.feature_importances_
    for feat, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
        print(f"    {feat:<40} {imp:.4f}")

    # ── Save models ───────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "isolation_forest.pkl"), "wb") as f:
        pickle.dump(iso, f)
    with open(os.path.join(MODEL_DIR, "random_forest.pkl"), "wb") as f:
        pickle.dump(rf, f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, "feature_cols.json"), "w") as f:
        json.dump(FEATURE_COLS, f)

    print(f"\nModels saved to {MODEL_DIR}/")
    print("Done.")


if __name__ == "__main__":
    np.random.seed(42)
    main()