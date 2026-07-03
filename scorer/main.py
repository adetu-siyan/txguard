import time
import random
import json
import os

from .generators import (
    generate_normal_transaction,
    generate_structuring_low,
    generate_structuring_medium,
    generate_structuring_high_cross_account,
    generate_atm_breach_variant,
    generate_tier1_breach_variant,
    generate_late_night_variant,
    generate_round_amount_sequence,
    generate_burst_sequence,
    generate_transfer_velocity_sequence,
)
from .eventhub_client import send_transaction, close_producer

OUTPUT_FILE = os.path.join(
    os.path.dirname(__file__), "..", "data", "simulated_transactions.jsonl"
)

# ── Population-Level Suspicious Rate ─────────────────────────────────────
# Calibrated against AMLNet (Huda et al., 2025), which targets ~0.16%
# laundering-positive against a backdrop noting real banks see roughly
# 1 in 21,000 transactions (~0.0048%). We deliberately run richer than
# either of those — small simulator runs need enough positive examples
# to be useful for training/demo purposes — but the principle holds:
# suspicious activity must stay RARE relative to normal traffic, or the
# model learns a fictional world where fraud is common.
SUSPICIOUS_RATE = 0.02  # ~2% of generated events are part of a suspicious pattern

# Within the suspicious bucket, relative likelihood of each pattern type.
# Structuring sophistication weighted toward LOW/MEDIUM (most realistic —
# most actual structuring attempts are unsophisticated), HIGH cross-account
# kept rarer since it represents a more deliberate, planned typology.
SUSPICIOUS_GENERATORS = [
    ("STRUCTURING_LOW", generate_structuring_low, 3, True),
    ("STRUCTURING_MEDIUM", generate_structuring_medium, 3, True),
    ("STRUCTURING_HIGH_CROSS_ACCOUNT", generate_structuring_high_cross_account, 1, True),
    ("ATM_BREACH", generate_atm_breach_variant, 4, False),
    ("TIER1_BREACH", generate_tier1_breach_variant, 3, False),
    ("LATE_NIGHT", generate_late_night_variant, 3, False),
    ("ROUND_AMOUNT", generate_round_amount_sequence, 2, True),
    ("BURST", generate_burst_sequence, 3, True),
    ("TRANSFER_VELOCITY", generate_transfer_velocity_sequence, 2, True),
]


def _write_and_send(txn, f):
    #send_transaction(txn)
    f.write(json.dumps(txn) + "\n")
    label = "SUSPICIOUS" if txn["is_suspicious"] else "normal"
    typ = txn.get("suspicious_typology") or "-"
    print(
        f"[{label:>10}] {txn['account_id']} ({txn['customer_id']}) "
        f"| {txn['account_tier']} | NGN {txn['amount']:,.2f} | {typ}"
    )


def generate_population_stream(count=5000):
    """
    Generates `count` transaction EVENTS (a sequence-producing generator
    counts as multiple events, one per transaction it returns) drawn from
    a realistic population: SUSPICIOUS_RATE of events come from an injected
    suspicious pattern, the remainder are archetype-driven normal traffic.

    This replaces the old fixed-count, fixed-anomaly-chance loop with a
    population model closer to how real transaction streams behave —
    mostly normal, rare and varied anomalies, with deliberate hard negatives
    mixed in so the boundary between normal and suspicious isn't trivial.
    """
    sent = 0
    weights = [w for _, _, w, _ in SUSPICIOUS_GENERATORS]
    generators = [(label, fn, is_seq) for label, fn, _, is_seq in SUSPICIOUS_GENERATORS]

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "a") as f:
        while sent < count:
            if random.random() < SUSPICIOUS_RATE:
                label, fn, is_sequence = random.choices(generators, weights=weights, k=1)[0]

                if is_sequence:
                    txns = fn()
                    for txn in txns:
                        if sent >= count:
                            break
                        _write_and_send(txn, f)
                        sent += 1
                        # time.sleep(random.uniform(0.3, 1.5))
                else:
                    txn = fn()
                    _write_and_send(txn, f)
                    sent += 1
                    # time.sleep(random.uniform(0.3, 1.5))
            else:
                txn = generate_normal_transaction()
                _write_and_send(txn, f)
                sent += 1
                # time.sleep(random.uniform(0.2, 1.0))


if __name__ == "__main__":
    try:
        # Larger default count than before (200 vs 20) — at a true ~2%
        # suspicious rate, a 20-event run would often produce zero or one
        # suspicious example, which isn't enough to validate anything against.
        generate_population_stream(count=5000)
    finally:
        close_producer()