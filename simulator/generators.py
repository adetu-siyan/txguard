import random
import uuid
from datetime import datetime, timedelta

from .config import (
    ACCOUNTS,
    ACCOUNT_TIERS,
    CHANNELS,
    LOCATIONS,
    CBN_LIMITS,
    ARCHETYPE_PROFILES,
    SUSPICIOUS_LABEL_FIELD,
    SUSPICIOUS_TYPE_FIELD,
    get_archetype,
    get_customer_id,
    get_customer_accounts,
)

account_balances = {acc: random.randint(20_000, 500_000) for acc in ACCOUNTS}


# ── Core Transaction Builder ─────────────────────────────────────────────
def _build_txn(account_id, amount, txn_type, channel,
                timestamp=None, is_suspicious=False, suspicious_typology=None):
    account_balances[account_id] = max(0, account_balances[account_id] - amount)

    ts = timestamp if timestamp is not None else datetime.utcnow()

    return {
        "transaction_id": f"TXN-{uuid.uuid4().hex[:8].upper()}",
        "account_id": account_id,
        "customer_id": get_customer_id(account_id),
        "account_tier": ACCOUNT_TIERS[account_id],
        "amount": round(amount, 2),
        "currency": "NGN",
        "type": txn_type,
        "channel": channel,
        "location": random.choice(LOCATIONS),
        "timestamp": ts.isoformat() + "Z",
        "balance_after": account_balances[account_id],
        # Ground-truth labels — for training/evaluation only, never for detection logic
        SUSPICIOUS_LABEL_FIELD: is_suspicious,
        SUSPICIOUS_TYPE_FIELD: suspicious_typology,
    }


def _random_timestamp(active_hours, weekend_factor=1.0, day_offset=0):
    """
    Produces a timestamp biased toward an archetype's active hours, with
    weekend activity scaled by weekend_factor. day_offset shifts the date
    backward/forward — used by suspicious sequences that span multiple days.
    """
    base_date = datetime.utcnow() - timedelta(days=day_offset)
    start_h, end_h = active_hours

    # Occasionally produce an off-hours transaction even for "normal" archetypes —
    # real people don't perfectly respect business hours. ~8% chance.
    if random.random() < 0.08:
        hour = random.randint(0, 23)
    else:
        hour = random.randint(start_h, end_h)

    return base_date.replace(
        hour=hour,
        minute=random.randint(0, 59),
        second=random.randint(0, 59),
        microsecond=0,
    )


# ── Normal Traffic (archetype-driven) ────────────────────────────────────
def generate_normal_transaction(account_id=None):
    """
    Produces one realistic transaction shaped by the account's customer
    archetype — amount range, channel preference, timing, and transaction
    type all vary by archetype rather than being flat-random.
    """
    if account_id is None:
        account_id = random.choice(ACCOUNTS)

    archetype = get_archetype(account_id)
    profile = ARCHETYPE_PROFILES[archetype]

    txn_type = random.choices(
        list(profile["txn_type_weights"].keys()),
        weights=list(profile["txn_type_weights"].values()),
        k=1,
    )[0]

    low, high = profile["amount_range"]
    # Most amounts cluster toward the lower-middle of the range (realistic —
    # people don't transact near their max every time). Use a triangular
    # distribution rather than uniform random.
    amount = random.triangular(low, high, low + (high - low) * 0.25)

    # Airtime and bills tend toward round-ish, smaller, more habitual amounts
    if txn_type == "airtime":
        amount = random.choice([100, 200, 500, 1_000, 2_000])
    elif txn_type == "bill_payment":
        amount = round(random.triangular(500, 20_000, 3_000), -1)

    channel = random.choice(profile["preferred_channels"])

    is_weekend = datetime.utcnow().weekday() >= 5
    weekend_factor = profile["weekend_activity_factor"]
    if is_weekend and random.random() > weekend_factor:
        # Skip-weight: lower-activity archetypes are less likely to transact
        # on weekends at all. We still return a transaction (the caller
        # controls overall volume), this just shifts the timestamp logic.
        pass

    ts = _random_timestamp(profile["active_hours"], weekend_factor)

    return _build_txn(account_id, amount, txn_type, channel, timestamp=ts)


# ── Suspicious Pattern Injection ──────────────────────────────────────────
# Adapted from AMLNet (Huda et al., 2025) sophistication-tier methodology,
# rescaled from AUSTRAC ($9,500/$10,000) thresholds to CBN thresholds
# (₦5,000,000 individual / ₦10,000,000 corporate, MLPPA 2022 Section 2).
#
# Three sophistication levels, matching the legal distinction we researched:
#   LOW / MEDIUM = single-account structuring (MLPPA Section 2(2))
#   HIGH         = cross-account coordinated structuring (same customer_id,
#                  multiple accounts) — the case static SQL rules cannot see.

def generate_structuring_low(account_id=None):
    """
    LOW sophistication: split into nearly-equal parts, each comfortably
    under the threshold, minimal variation. The crudest, most obvious form —
    an unsophisticated actor splitting one large sum into 2-4 similar pieces.
    """
    if account_id is None:
        account_id = random.choice(ACCOUNTS)

    threshold = CBN_LIMITS["NFIU_INDIVIDUAL_THRESHOLD"]
    total = random.uniform(threshold * 1.05, threshold * 1.8)
    num_splits = random.randint(2, 4)

    base_part = total / num_splits
    txns = []
    for i in range(num_splits):
        variation = random.uniform(-0.05, 0.05)  # ±5% noise only
        amount = base_part * (1 + variation)
        amount = min(amount, threshold * 0.98)  # never exceed threshold

        ts = _random_timestamp((8, 20), day_offset=0) + timedelta(
            hours=i * random.uniform(0.5, 3)
        )
        txns.append(_build_txn(
            account_id, amount, "transfer", random.choice(["mobile", "USSD"]),
            timestamp=ts, is_suspicious=True, suspicious_typology="structuring_low",
        ))
    return txns


def generate_structuring_medium(account_id=None):
    """
    MEDIUM sophistication: splits drawn from a normal distribution around
    the mean part size, all clipped below threshold. More natural-looking
    variation than LOW, spread across a wider time window (same day to
    next day) to look less mechanical.
    """
    if account_id is None:
        account_id = random.choice(ACCOUNTS)

    threshold = CBN_LIMITS["NFIU_INDIVIDUAL_THRESHOLD"]
    total = random.uniform(threshold * 1.1, threshold * 2.2)
    num_splits = random.randint(3, 6)

    mean_part = total / num_splits
    std_dev = mean_part * 0.18

    txns = []
    for i in range(num_splits):
        amount = max(1_000, random.gauss(mean_part, std_dev))
        amount = min(amount, threshold * 0.97)

        day_offset = 0 if i < num_splits // 2 else random.choice([0, 1])
        ts = _random_timestamp((7, 21), day_offset=day_offset) + timedelta(
            hours=random.uniform(0, 6)
        )
        txns.append(_build_txn(
            account_id, amount, "transfer", random.choice(["mobile", "USSD", "POS"]),
            timestamp=ts, is_suspicious=True, suspicious_typology="structuring_medium",
        ))
    return txns


def generate_structuring_high_cross_account():
    """
    HIGH sophistication: the actual cross-account case. Picks a customer
    with 2+ accounts, splits a large sum across them using a lognormal
    distribution (a few transactions just under threshold, the rest smaller
    and more varied), with timestamps spread across ±2 days and varied
    channels — designed to look like unrelated activity unless you can
    see the shared customer_id.

    This is the canonical example a per-account SQL rule structurally
    cannot catch, and the primary justification for the GNN layer.
    """
    multi_account_customers = [
        cust for cust in {get_customer_id(a) for a in ACCOUNTS}
        if len(get_customer_accounts(cust)) >= 2
    ]

    if not multi_account_customers:
        # Fallback: no multi-account customer exists in this run: degrade
        # gracefully to medium single-account structuring rather than crash.
        return generate_structuring_medium()

    customer_id = random.choice(multi_account_customers)
    customer_accounts = get_customer_accounts(customer_id)

    threshold = CBN_LIMITS["NFIU_INDIVIDUAL_THRESHOLD"]
    total = random.uniform(threshold * 1.3, threshold * 3.0)

    num_splits = random.randint(4, 9)
    # Lognormal distribution for the bulk of splits — produces a realistic
    # right-skewed spread (many smaller amounts, a few larger ones)
    raw_splits = [random.lognormvariate(mu=0, sigma=0.6) for _ in range(num_splits)]
    raw_total = sum(raw_splits)
    splits = [total * (r / raw_total) for r in raw_splits]

    # Force 1-2 splits to sit just under the threshold (the AMLNet "high"
    # signature — not everything is disguised, a couple are bold)
    num_near_threshold = random.randint(1, 2)
    for i in range(num_near_threshold):
        splits[i] = threshold * random.uniform(0.92, 0.99)

    txns = []
    for amount in splits:
        amount = min(amount, threshold * 0.99)
        account_id = random.choice(customer_accounts)
        day_offset = random.randint(0, 2)
        ts = _random_timestamp((6, 23), day_offset=day_offset)
        channel = random.choice(["mobile", "USSD", "POS", "ATM"])
        txn_type = random.choice(["transfer", "withdrawal"])

        txns.append(_build_txn(
            account_id, amount, txn_type, channel,
            timestamp=ts, is_suspicious=True,
            suspicious_typology="structuring_high_cross_account",
        ))

    return txns


# ── CBN Rule-Boundary Generators (probabilistic, not deterministic) ──────
# These replace the old "always trip the rule cleanly" generators. Amounts
# now vary across a range that includes clear violations, borderline cases,
# and near-misses, so a model has to learn the actual signal rather than
# memorize a fixed template.

def generate_atm_breach_variant():
    """
    CBN Revised Cash-Related Policies Circular — ATM daily cap ₦100,000.
    Produces violations ranging from barely-over to grossly-over, plus
    occasional near-misses just under the cap (hard negatives — important
    for the model to learn the boundary precisely, not just "big = bad").
    """
    account_id = random.choice(ACCOUNTS)
    limit = CBN_LIMITS["ATM_DAILY_MAX"]

    roll = random.random()
    if roll < 0.15:
        # Hard negative: just under the limit, NOT suspicious
        amount = random.uniform(limit * 0.85, limit * 0.99)
        is_suspicious = False
        typology = None
    elif roll < 0.55:
        # Borderline breach
        amount = random.uniform(limit * 1.01, limit * 1.3)
        is_suspicious = True
        typology = "atm_breach_borderline"
    else:
        # Clear breach
        amount = random.uniform(limit * 1.3, limit * 3.0)
        is_suspicious = True
        typology = "atm_breach_clear"

    return _build_txn(
        account_id, amount, "withdrawal", "ATM",
        timestamp=_random_timestamp((7, 22)),
        is_suspicious=is_suspicious, suspicious_typology=typology,
    )


def generate_tier1_breach_variant():
    """
    CBN Tiered KYC Framework — Tier 1 daily cap ₦30,000.
    """
    tier1_accounts = [acc for acc, tier in ACCOUNT_TIERS.items() if tier == "TIER_1"]
    if not tier1_accounts:
        return generate_normal_transaction()

    account_id = random.choice(tier1_accounts)
    limit = CBN_LIMITS["TIER_1_DAILY_MAX"]

    roll = random.random()
    if roll < 0.2:
        amount = random.uniform(limit * 0.7, limit * 0.98)
        is_suspicious = False
        typology = None
    else:
        amount = random.uniform(limit * 1.05, limit * 4.0)
        is_suspicious = True
        typology = "kyc_tier1_breach"

    return _build_txn(
        account_id, amount, "transfer", random.choice(["mobile", "USSD"]),
        timestamp=_random_timestamp((8, 21)),
        is_suspicious=is_suspicious, suspicious_typology=typology,
    )


def generate_late_night_variant():
    """
    CBN CDD Regulations 2023 — transactions inconsistent with customer
    risk profile. True signal is the COMBINATION of late-night timing AND
    high value, not either alone — so we generate all four quadrants.
    """
    account_id = random.choice(ACCOUNTS)
    archetype = get_archetype(account_id)

    # gig_worker genuinely transacts at all hours — late night alone isn't
    # anomalous for them, so we down-weight suspicion for that archetype
    is_late_hour = random.random() < 0.5
    is_high_value = random.random() < 0.5

    hour = random.randint(
        CBN_LIMITS["LATE_NIGHT_HOUR_START"], CBN_LIMITS["LATE_NIGHT_HOUR_END"]
    ) if is_late_hour else random.randint(8, 20)

    amount = (
        random.uniform(CBN_LIMITS["LATE_NIGHT_MIN_AMOUNT"], 300_000)
        if is_high_value else random.uniform(1_000, CBN_LIMITS["LATE_NIGHT_MIN_AMOUNT"] * 0.9)
    )

    is_suspicious = is_late_hour and is_high_value and archetype != "gig_worker"
    typology = "late_night_high_value" if is_suspicious else None

    ts = datetime.utcnow().replace(
        hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59)
    )

    return _build_txn(
        account_id, amount, "withdrawal", "ATM",
        timestamp=ts, is_suspicious=is_suspicious, suspicious_typology=typology,
    )


def generate_round_amount_sequence():
    """
    NFIU/FATF layering typology — round-number transfers are statistically
    unusual for genuine commerce. Mix genuine round-number sequences
    (suspicious) with the occasional coincidental round number in normal
    traffic (hard negative — a single round amount alone proves nothing).
    """
    account_id = random.choice(ACCOUNTS)
    num_txns = random.randint(2, 5)
    is_suspicious_sequence = num_txns >= 3  # 2 round numbers is just coincidence

    txns = []
    for i in range(num_txns):
        amount = random.choice([10_000, 20_000, 50_000, 100_000, 150_000, 200_000])
        ts = _random_timestamp((8, 21), day_offset=0) + timedelta(hours=i * random.uniform(1, 5))
        txns.append(_build_txn(
            account_id, amount, "transfer", random.choice(["mobile", "USSD"]),
            timestamp=ts,
            is_suspicious=is_suspicious_sequence,
            suspicious_typology="round_amount_pattern" if is_suspicious_sequence else None,
        ))
    return txns


def generate_burst_sequence():
    """
    Account takeover / fraud velocity signal — 3+ rapid transactions within
    60 seconds. Includes occasional 2-transaction near-misses as hard negatives.
    """
    account_id = random.choice(ACCOUNTS)
    num_txns = random.choice([2, 3, 3, 4, 5])  # weighted toward 3 (the rule threshold)
    is_suspicious = num_txns >= 3

    base_ts = datetime.utcnow()
    txns = []
    for i in range(num_txns):
        ts = base_ts + timedelta(seconds=random.uniform(0, 55))
        amount = random.uniform(5_000, 60_000)
        txns.append(_build_txn(
            account_id, amount, random.choice(["transfer", "withdrawal"]),
            random.choice(CHANNELS), timestamp=ts,
            is_suspicious=is_suspicious,
            suspicious_typology="burst_pattern" if is_suspicious else None,
        ))
    return txns


def generate_transfer_velocity_sequence():
    """
    MLPPA 2022 structuring-adjacent — many small transfers in a short window,
    each individually unremarkable.
    """
    account_id = random.choice(ACCOUNTS)
    num_txns = random.choice([3, 4, 5, 5, 6, 7])
    is_suspicious = num_txns >= 5

    base_ts = datetime.utcnow()
    txns = []
    for i in range(num_txns):
        ts = base_ts + timedelta(minutes=random.uniform(0, 50))
        amount = random.uniform(8_000, 95_000)
        txns.append(_build_txn(
            account_id, amount, "transfer", random.choice(["mobile", "USSD"]),
            timestamp=ts,
            is_suspicious=is_suspicious,
            suspicious_typology="transfer_velocity" if is_suspicious else None,
        ))
    return txns