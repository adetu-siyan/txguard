-- ============================================================
-- TxGuard Stream Analytics Rules — TEST MODE
-- ⚠️  FOR DEMO/VALIDATION ONLY — DO NOT USE IN PRODUCTION
--
-- Changes from production (rule_prod.sql):
--   • All windows compressed to 2 minutes (from 24hr / 7day / 1hr)
--   • Burst window kept at 60 seconds (already short — unchanged)
--   • All HAVING thresholds lowered to fire within a single simulator run
--   • Late Night rule: hour filter removed (fires any time of day)
--   • Rule 6 timestamp parsing note added (see inline comment)
--
-- To restore production: swap windows and thresholds back per comments
-- Input alias:  transactions
-- Output alias: FlaggedTransactions
-- ============================================================


-- ─── RULE 1: Daily ATM Breach ─────────────────────────────
-- PROD:  TumblingWindow(hour, 24) | HAVING SUM(amount) > 100000
-- TEST:  TumblingWindow(minute, 2) | HAVING SUM(amount) > 1000

SELECT
    account_id,
    account_tier,
    SUM(amount) AS flagged_amount,
    COUNT(*) AS txn_count,
    'DAILY_ATM_BREACH' AS rule_triggered,
    'HIGH' AS severity,
    System.Timestamp() AS detected_at
INTO [FlaggedTransactions]
FROM [transactions] TIMESTAMP BY timestamp
WHERE channel = 'ATM' AND type = 'withdrawal'
GROUP BY account_id, account_tier, TumblingWindow(minute, 2)
HAVING SUM(amount) > 1000;


-- ─── RULE 2: Weekly Individual Withdrawal Breach ───────────
-- PROD:  TumblingWindow(day, 7) | HAVING SUM(amount) > 500000
-- TEST:  TumblingWindow(minute, 2) | HAVING SUM(amount) > 2000

SELECT
    account_id,
    account_tier,
    SUM(amount) AS flagged_amount,
    COUNT(*) AS txn_count,
    'WEEKLY_WITHDRAWAL_BREACH' AS rule_triggered,
    'HIGH' AS severity,
    System.Timestamp() AS detected_at
INTO [FlaggedTransactions]
FROM [transactions] TIMESTAMP BY timestamp
WHERE type = 'withdrawal'
GROUP BY account_id, account_tier, TumblingWindow(minute, 2)
HAVING SUM(amount) > 2000;


-- ─── RULE 3: Burst Pattern ────────────────────────────────
-- PROD:  TumblingWindow(second, 60) | HAVING COUNT(*) >= 3
-- TEST:  UNCHANGED — window already short enough to fire during simulator run

SELECT
    account_id,
    account_tier,
    COUNT(*) AS txn_count,
    SUM(amount) AS flagged_amount,
    'BURST_PATTERN' AS rule_triggered,
    'HIGH' AS severity,
    System.Timestamp() AS detected_at
INTO [FlaggedTransactions]
FROM [transactions] TIMESTAMP BY timestamp
GROUP BY account_id, account_tier, TumblingWindow(second, 60)
HAVING COUNT(*) >= 3;


-- ─── RULE 4: Structuring Detection ───────────────────────
-- PROD:  TumblingWindow(hour, 24) | amount >= 90000 AND <= 99999 | HAVING COUNT(*) >= 3
-- TEST:  TumblingWindow(minute, 2) | amount >= 900 AND <= 99999 | HAVING COUNT(*) >= 2

SELECT
    account_id,
    account_tier,
    COUNT(*) AS txn_count,
    SUM(amount) AS flagged_amount,
    'STRUCTURING_DETECTED' AS rule_triggered,
    'HIGH' AS severity,
    System.Timestamp() AS detected_at
INTO [FlaggedTransactions]
FROM [transactions] TIMESTAMP BY timestamp
WHERE channel = 'ATM'
  AND type = 'withdrawal'
  AND amount >= 900
  AND amount <= 99999
GROUP BY account_id, account_tier, TumblingWindow(minute, 2)
HAVING COUNT(*) >= 2;


-- ─── RULE 5: NFIU Reporting Threshold Approach ───────────
-- PROD:  TumblingWindow(hour, 24) | HAVING SUM(amount) > 4500000
-- TEST:  TumblingWindow(minute, 2) | HAVING SUM(amount) > 5000

SELECT
    account_id,
    account_tier,
    SUM(amount) AS flagged_amount,
    COUNT(*) AS txn_count,
    'NFIU_THRESHOLD_APPROACH' AS rule_triggered,
    'HIGH' AS severity,
    System.Timestamp() AS detected_at
INTO [FlaggedTransactions]
FROM [transactions] TIMESTAMP BY timestamp
WHERE type = 'transfer' OR type = 'withdrawal'
GROUP BY account_id, account_tier, TumblingWindow(minute, 2)
HAVING SUM(amount) > 5000;


-- ─── RULE 6: Late Night High-Value Withdrawal ─────────────
-- PROD:  DATEPART(hour) >= 1 AND <= 4 | amount > 50000
-- TEST:  Hour filter REMOVED (so it fires any time of day) | amount > 500
-- NOTE:  If DATEPART(hour, timestamp) returns wrong values, your simulator
--        timestamp string may not be parsing correctly. Check that
--        datetime.utcnow().isoformat() + "Z" is being sent — it should be
--        fine, but verify in Event Hubs metrics that arrival time ≈ event time.

SELECT
    account_id,
    account_tier,
    amount AS flagged_amount,
    1 AS txn_count,
    'LATE_NIGHT_HIGH_VALUE' AS rule_triggered,
    'MEDIUM' AS severity,
    System.Timestamp() AS detected_at
INTO [FlaggedTransactions]
FROM [transactions] TIMESTAMP BY timestamp
WHERE type = 'withdrawal'
  AND amount > 500;


-- ─── RULE 7: Transfer Velocity ────────────────────────────
-- PROD:  TumblingWindow(hour, 1) | HAVING COUNT(*) >= 5 AND AVG < 100000
-- TEST:  TumblingWindow(minute, 2) | HAVING COUNT(*) >= 2 AND AVG < 100000

SELECT
    account_id,
    account_tier,
    COUNT(*) AS txn_count,
    SUM(amount) AS flagged_amount,
    'TRANSFER_VELOCITY' AS rule_triggered,
    'MEDIUM' AS severity,
    System.Timestamp() AS detected_at
INTO [FlaggedTransactions]
FROM [transactions] TIMESTAMP BY timestamp
WHERE type = 'transfer'
GROUP BY account_id, account_tier, TumblingWindow(minute, 2)
HAVING COUNT(*) >= 2
  AND AVG(CAST(amount AS float)) < 100000;


-- ─── RULE 8: KYC Tier 1 Daily Limit Breach ───────────────
-- PROD:  TumblingWindow(hour, 24) | HAVING SUM(amount) > 30000
-- TEST:  TumblingWindow(minute, 2) | HAVING SUM(amount) > 500

SELECT
    account_id,
    account_tier,
    SUM(amount) AS flagged_amount,
    COUNT(*) AS txn_count,
    'KYC_TIER1_BREACH' AS rule_triggered,
    'HIGH' AS severity,
    System.Timestamp() AS detected_at
INTO [FlaggedTransactions]
FROM [transactions] TIMESTAMP BY timestamp
WHERE account_tier = 'TIER_1'
GROUP BY account_id, account_tier, TumblingWindow(minute, 2)
HAVING SUM(amount) > 500;


-- ─── RULE 10: Round Amount Transfer Pattern ───────────────
-- PROD:  TumblingWindow(hour, 24) | HAVING COUNT(*) >= 3
-- TEST:  TumblingWindow(minute, 2) | HAVING COUNT(*) >= 2

SELECT
    account_id,
    account_tier,
    COUNT(*) AS txn_count,
    SUM(amount) AS flagged_amount,
    'ROUND_AMOUNT_PATTERN' AS rule_triggered,
    'MEDIUM' AS severity,
    System.Timestamp() AS detected_at
INTO [FlaggedTransactions]
FROM [transactions] TIMESTAMP BY timestamp
WHERE type = 'transfer'
  AND CAST(amount AS bigint) % 10000 = 0
GROUP BY account_id, account_tier, TumblingWindow(minute, 2)
HAVING COUNT(*) >= 2;
