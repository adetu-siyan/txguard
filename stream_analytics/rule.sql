-- ============================================================
-- TxGuard Stream Analytics Rules
-- Sources: CBN Jan 2026 Cash Circular, MLPPA 2022,
--          CBN CDD Regulations 2023, CBN TKYC Framework
-- Input alias: transactions
-- Output alias: FlaggedTransactions
-- ============================================================


-- ─── RULE 1: Daily ATM Breach ─────────────────────────────
-- Source: CBN Revised Cash-Related Policies Circular Jan 2026
-- Threshold: ₦100,000 per customer per day via ATM

SELECT
    account_id,
    account_tier,
    SUM(amount)             AS flagged_amount,
    COUNT(*)                AS txn_count,
    'DAILY_ATM_BREACH'      AS rule_triggered,
    'HIGH'                  AS severity,
    System.Timestamp()      AS detected_at
INTO [FlaggedTransactions]
FROM [transactions] TIMESTAMP BY timestamp
WHERE channel = 'ATM' AND type = 'withdrawal'
GROUP BY account_id, account_tier, TumblingWindow(hour, 24)
HAVING SUM(amount) > 100000;


-- ─── RULE 2: Weekly Individual Withdrawal Breach ───────────
-- Source: CBN Revised Cash-Related Policies Circular Jan 2026
-- Threshold: ₦500,000 per individual per week across all channels

SELECT
    account_id,
    account_tier,
    SUM(amount)                     AS flagged_amount,
    COUNT(*)                        AS txn_count,
    'WEEKLY_WITHDRAWAL_BREACH'      AS rule_triggered,
    'HIGH'                          AS severity,
    System.Timestamp()              AS detected_at
INTO [FlaggedTransactions]
FROM [transactions] TIMESTAMP BY timestamp
WHERE type = 'withdrawal'
GROUP BY account_id, account_tier, TumblingWindow(day, 7)
HAVING SUM(amount) > 500000;


-- ─── RULE 3: Burst Pattern ────────────────────────────────
-- Source: MLPPA 2022 structuring prohibition + account takeover typology
-- 3+ transactions from same account within 60 seconds

SELECT
    account_id,
    account_tier,
    COUNT(*)            AS txn_count,
    SUM(amount)         AS flagged_amount,
    'BURST_PATTERN'     AS rule_triggered,
    'HIGH'              AS severity,
    System.Timestamp()  AS detected_at
INTO [FlaggedTransactions]
FROM [transactions] TIMESTAMP BY timestamp
GROUP BY account_id, account_tier, TumblingWindow(second, 60)
HAVING COUNT(*) >= 3;


-- ─── RULE 4: Structuring Detection ───────────────────────
-- Source: MLPPA 2022 — splitting transactions to evade limits is prohibited
-- 3+ ATM withdrawals in ₦90,000–₦99,999 range within 24 hours

SELECT
    account_id,
    account_tier,
    COUNT(*)                AS txn_count,
    SUM(amount)             AS flagged_amount,
    'STRUCTURING_DETECTED'  AS rule_triggered,
    'HIGH'                  AS severity,
    System.Timestamp()      AS detected_at
INTO [FlaggedTransactions]
FROM [transactions] TIMESTAMP BY timestamp
WHERE channel = 'ATM'
  AND type = 'withdrawal'
  AND amount >= 90000
  AND amount <= 99999
GROUP BY account_id, account_tier, TumblingWindow(hour, 24)
HAVING COUNT(*) >= 3;


-- ─── RULE 5: NFIU Reporting Threshold Approach ───────────
-- Source: MLPPA 2022 — ₦5M individual threshold triggers mandatory NFIU report
-- Flag at ₦4.5M (90%) so compliance team can act before obligation fires

SELECT
    account_id,
    account_tier,
    SUM(amount)                     AS flagged_amount,
    COUNT(*)                        AS txn_count,
    'NFIU_THRESHOLD_APPROACH'       AS rule_triggered,
    'HIGH'                          AS severity,
    System.Timestamp()              AS detected_at
INTO [FlaggedTransactions]
FROM [transactions] TIMESTAMP BY timestamp
WHERE type = 'transfer' OR type = 'withdrawal'
GROUP BY account_id, account_tier, TumblingWindow(hour, 24)
HAVING SUM(amount) > 4500000;


-- ─── RULE 6: Late Night High-Value Withdrawal ─────────────
-- Source: CBN CDD Regulations 2023 — transactions must match customer risk profile
-- Withdrawal > ₦50,000 between 1am and 4am

SELECT
    account_id,
    account_tier,
    amount                      AS flagged_amount,
    1                           AS txn_count,
    'LATE_NIGHT_HIGH_VALUE'     AS rule_triggered,
    'MEDIUM'                    AS severity,
    System.Timestamp()          AS detected_at
INTO [FlaggedTransactions]
FROM [transactions] TIMESTAMP BY timestamp
WHERE type = 'withdrawal'
  AND amount > 50000
  AND DATEPART(hour, timestamp) >= 1
  AND DATEPART(hour, timestamp) <= 4;


-- ─── RULE 7: Transfer Velocity ────────────────────────────
-- Source: MLPPA 2022 structuring prohibition
-- 5+ transfers within 1 hour, each individually small

SELECT
    account_id,
    account_tier,
    COUNT(*)                AS txn_count,
    SUM(amount)             AS flagged_amount,
    'TRANSFER_VELOCITY'     AS rule_triggered,
    'MEDIUM'                AS severity,
    System.Timestamp()      AS detected_at
INTO [FlaggedTransactions]
FROM [transactions] TIMESTAMP BY timestamp
WHERE type = 'transfer'
GROUP BY account_id, account_tier, TumblingWindow(hour, 1)
HAVING COUNT(*) >= 5
   AND AVG(CAST(amount AS float)) < 100000;


-- ─── RULE 8: KYC Tier 1 Daily Limit Breach ───────────────
-- Source: CBN Tiered KYC Framework
-- Tier 1 accounts (BVN/NIN only) capped at ₦30,000 per day

SELECT
    account_id,
    account_tier,
    SUM(amount)         AS flagged_amount,
    COUNT(*)            AS txn_count,
    'KYC_TIER1_BREACH'  AS rule_triggered,
    'HIGH'              AS severity,
    System.Timestamp()  AS detected_at
INTO [FlaggedTransactions]
FROM [transactions] TIMESTAMP BY timestamp
WHERE account_tier = 'TIER_1'
GROUP BY account_id, account_tier, TumblingWindow(hour, 24)
HAVING SUM(amount) > 30000;


-- ─── RULE 10: Round Amount Transfer Pattern ───────────────
-- Source: NFIU/FATF money laundering typologies — layering detection
-- 3+ transfers of exact round amounts (multiples of ₦10,000) in 24 hours

SELECT
    account_id,
    account_tier,
    COUNT(*)                AS txn_count,
    SUM(amount)             AS flagged_amount,
    'ROUND_AMOUNT_PATTERN'  AS rule_triggered,
    'MEDIUM'                AS severity,
    System.Timestamp()      AS detected_at
INTO [FlaggedTransactions]
FROM [transactions] TIMESTAMP BY timestamp
WHERE type = 'transfer'
  AND CAST(amount AS bigint) % 10000 = 0
GROUP BY account_id, account_tier, TumblingWindow(hour, 24)
HAVING COUNT(*) >= 3;