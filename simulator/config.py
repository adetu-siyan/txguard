import random

# ── Customer & Account Model ────────────────────────────────────────────
# Real customers can hold 2-3 accounts. This is what makes cross-account
# structuring detection possible — without a shared customer_id, no system
# (rule-based or ML) can see that two "different" accounts are the same person.

NUM_CUSTOMERS = 18  # fewer customers than accounts, since some hold multiple

CUSTOMER_IDS = [f"CUST-{2000 + i}" for i in range(NUM_CUSTOMERS)]

# Assign each customer 1-3 accounts. Most customers have 1 account (realistic),
# a minority have 2-3 (also realistic — savings + current, or personal + small business).
def _build_customer_accounts():
    mapping = {}  # account_id -> customer_id
    accounts = []
    acc_counter = 1000

    for cust in CUSTOMER_IDS:
        num_accounts = random.choices([1, 2, 3], weights=[70, 22, 8], k=1)[0]
        for _ in range(num_accounts):
            acc_id = f"ACC-{acc_counter}"
            mapping[acc_id] = cust
            accounts.append(acc_id)
            acc_counter += 1

    return accounts, mapping

ACCOUNTS, ACCOUNT_TO_CUSTOMER = _build_customer_accounts()

CHANNELS = ["ATM", "mobile", "POS", "USSD"]
TXN_TYPES = ["transfer", "withdrawal", "bill_payment", "airtime"]
LOCATIONS = ["Lagos", "Abuja", "Ibadan", "Port Harcourt", "Kano"]

# ── CBN Tiered KYC ───────────────────────────────────────────────────────
# Tier is assigned per CUSTOMER, not per account — a person's KYC tier
# doesn't change depending on which of their accounts they're using.
ACCOUNT_TIERS = {}
for cust in CUSTOMER_IDS:
    tier = random.choices(
        ["TIER_1", "TIER_2", "TIER_3"], weights=[40, 40, 20], k=1
    )[0]
    for acc in ACCOUNTS:
        if ACCOUNT_TO_CUSTOMER[acc] == cust:
            ACCOUNT_TIERS[acc] = tier

# ── Customer Archetypes ──────────────────────────────────────────────────
# Drives realistic, differentiated "normal" behavior. Without this, every
# normal transaction looks statistically identical and a model can't learn
# what "normal" even means for a given customer type.
#
# salary_earner   — regular monthly inflow, steady bill payments, low velocity
# small_trader    — frequent POS/transfer activity, irregular amounts, business hours
# gig_worker      — frequent small inflows/outflows, mobile-heavy, all hours
# low_activity    — sparse transactions, mostly airtime/bills, occasional withdrawal

ARCHETYPES = ["salary_earner", "small_trader", "gig_worker", "low_activity"]
ARCHETYPE_WEIGHTS = [35, 30, 20, 15]

CUSTOMER_ARCHETYPE = {
    cust: random.choices(ARCHETYPES, weights=ARCHETYPE_WEIGHTS, k=1)[0]
    for cust in CUSTOMER_IDS
}

def get_archetype(account_id):
    cust = ACCOUNT_TO_CUSTOMER[account_id]
    return CUSTOMER_ARCHETYPE[cust]

def get_customer_id(account_id):
    return ACCOUNT_TO_CUSTOMER[account_id]

def get_customer_accounts(customer_id):
    return [acc for acc, cust in ACCOUNT_TO_CUSTOMER.items() if cust == customer_id]

# ── Per-Archetype Behavioral Profiles ────────────────────────────────────
# Used by the normal-traffic generator to produce differentiated, realistic
# transaction patterns instead of flat random amounts.

ARCHETYPE_PROFILES = {
    "salary_earner": {
        "txn_type_weights": {"transfer": 30, "bill_payment": 35, "airtime": 20, "withdrawal": 15},
        "amount_range": (1_000, 80_000),
        "preferred_channels": ["mobile", "USSD"],
        "active_hours": (7, 21),       # rarely transacts late night
        "weekend_activity_factor": 0.5, # quieter on weekends
        "daily_txn_count_range": (1, 4),
    },
    "small_trader": {
        "txn_type_weights": {"transfer": 25, "withdrawal": 30, "bill_payment": 15, "airtime": 30},
        "amount_range": (2_000, 250_000),
        "preferred_channels": ["POS", "ATM", "mobile"],
        "active_hours": (7, 22),
        "weekend_activity_factor": 1.1,  # markets often busier on weekends
        "daily_txn_count_range": (3, 10),
    },
    "gig_worker": {
        "txn_type_weights": {"transfer": 40, "airtime": 25, "bill_payment": 15, "withdrawal": 20},
        "amount_range": (500, 60_000),
        "preferred_channels": ["mobile", "USSD"],
        "active_hours": (0, 23),  # genuinely irregular hours, not a red flag for this archetype
        "weekend_activity_factor": 1.0,
        "daily_txn_count_range": (2, 7),
    },
    "low_activity": {
        "txn_type_weights": {"airtime": 40, "bill_payment": 35, "transfer": 15, "withdrawal": 10},
        "amount_range": (200, 30_000),
        "preferred_channels": ["USSD", "mobile"],
        "active_hours": (8, 20),
        "weekend_activity_factor": 0.6,
        "daily_txn_count_range": (0, 2),
    },
}

# ── CBN-Grounded Thresholds ───────────────────────────────────────────────
# Sources cited per field. These are unchanged in meaning from before, but
# now feed a probabilistic injection model rather than deterministic templates.
CBN_LIMITS = {
    # CBN Revised Cash-Related Policies Circular, effective Jan 1 2026
    "ATM_DAILY_MAX": 100_000,
    "WEEKLY_INDIVIDUAL_MAX": 500_000,
    "WEEKLY_CORPORATE_MAX": 5_000_000,

    # MLPPA 2022, Section 2(1) — single transaction reporting trigger
    "NFIU_INDIVIDUAL_THRESHOLD": 5_000_000,
    "NFIU_CORPORATE_THRESHOLD": 10_000_000,

    # CBN Tiered KYC Framework
    "TIER_1_DAILY_MAX": 30_000,
    "TIER_2_DAILY_MAX": 500_000,

    # CBN CDD Regulations 2023 — profile consistency / late-night red flag
    "LATE_NIGHT_MIN_AMOUNT": 50_000,
    "LATE_NIGHT_HOUR_START": 1,
    "LATE_NIGHT_HOUR_END": 4,
}

# ── Synthetic Label Ground Truth ──────────────────────────────────────────
# is_suspicious is for TRAINING/EVALUATION ONLY. It must never be passed into
# any detection logic (rules or ML) — that would be label leakage. It exists
# so we can later measure whether the detection layer actually catches what
# we deliberately injected.
SUSPICIOUS_LABEL_FIELD = "is_suspicious"
SUSPICIOUS_TYPE_FIELD = "suspicious_typology"   # e.g. "structuring_low", "structuring_high_cross_account"