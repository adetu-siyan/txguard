import pandas as pd
import numpy as np
from collections import defaultdict

df = pd.read_csv('data/features.csv')

accounts = df['account_id'].unique().tolist()
account_to_idx = {acc: i for i, acc in enumerate(accounts)}
num_nodes = len(accounts)

FEATURE_COLS = [
    'amount', 'amount_log', 'threshold_proximity', 'amount_roundness',
    'is_near_threshold', 'exceeds_atm_daily', 'exceeds_tier1_daily',
    'hour_of_day', 'is_late_night', 'is_weekend',
    'inter_txn_interval_seconds', 'daily_txn_count', 'hourly_txn_count',
    'daily_velocity', 'amount_vs_customer_mean', 'amount_zscore',
    'channel_consistency', 'hour_consistency',
    'sum_1h', 'sum_24h', 'sum_7d', 'count_1h', 'count_24h',
    'cross_account_sum_24h', 'cross_account_threshold_ratio',
    'cross_account_sum_6h', 'cross_account_ratio_6h',
    'max_single_24h', 'cov_24h',
    'accounts_per_customer', 'account_age_days',
    'unique_counterparties_24h', 'tier_numeric',
    'channel_numeric', 'type_numeric',
]

node_features = np.zeros((num_nodes, len(FEATURE_COLS)))
node_txn_counts = np.zeros(num_nodes)

for _, row in df.iterrows():
    acc = row['account_id']
    idx = account_to_idx[acc]
    node_features[idx] += np.array([float(row[c]) for c in FEATURE_COLS])
    node_txn_counts[idx] += 1

for i in range(num_nodes):
    if node_txn_counts[i] > 0:
        node_features[i] /= node_txn_counts[i]

print('Node feature matrix shape:', node_features.shape)
print('NaN after aggregation:', np.isnan(node_features).sum())
print('Inf after aggregation:', np.isinf(node_features).sum())

stds = node_features.std(axis=0)
print()
print('Columns with zero std across nodes:')
zero_std_found = False
for i, (col, std) in enumerate(zip(FEATURE_COLS, stds)):
    if std == 0:
        print('  ', col, 'std=0, unique values=', len(np.unique(node_features[:, i])))
        zero_std_found = True
if not zero_std_found:
    print('  None — all columns have non-zero std')

print()
print('Num accounts:', num_nodes)

suspicious_per_account = df.groupby('account_id')['is_suspicious'].max()
print('Suspicious accounts:', int((suspicious_per_account == 1).sum()))

customer_to_accounts = defaultdict(list)
for _, row in df.drop_duplicates('account_id').iterrows():
    cust = row.get('customer_id')
    acc = row['account_id']
    if cust:
        customer_to_accounts[str(cust)].append(acc)

multi = {c: accs for c, accs in customer_to_accounts.items() if len(accs) > 1}
print('Customers with multiple accounts:', len(multi))
total_edges = sum(len(a) * (len(a) - 1) for a in multi.values())
print('Total cross-account edge pairs:', total_edges)

print()
print('Feature value ranges (post-aggregation):')
for i, col in enumerate(FEATURE_COLS):
    col_data = node_features[:, i]
    print(f'  {col:<40} min={col_data.min():>10.3f}  max={col_data.max():>10.3f}  std={stds[i]:>8.4f}')