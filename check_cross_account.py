import csv
rows = list(csv.DictReader(open('data/features.csv')))
suspicious = [r for r in rows if r['is_suspicious'] == '1']
cross = [r for r in suspicious if 'cross_account' in r.get('suspicious_typology', '')]
single = [r for r in suspicious if 'structuring' in r.get('suspicious_typology', '') and 'cross' not in r.get('suspicious_typology', '')]

print('Cross-account structuring events:')
for r in cross:
    print(f"  {r['account_id']} ({r['customer_id']}) | amount={float(r['amount']):,.0f} | cross_sum_24h={float(r['cross_account_sum_24h']):,.0f} | ratio={float(r['cross_account_threshold_ratio']):.2f}")

print()
print('Single-account structuring events:')
for r in single[:5]:
    print(f"  {r['account_id']} ({r['customer_id']}) | amount={float(r['amount']):,.0f} | cross_sum_24h={float(r['cross_account_sum_24h']):,.0f} | ratio={float(r['cross_account_threshold_ratio']):.2f}")