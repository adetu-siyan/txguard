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

SUSPICIOUS_RATE = 0.02

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
    # send_transaction(txn)  # disabled for offline run
    f.write(json.dumps(txn) + "\n")
    label = "SUSPICIOUS" if txn["is_suspicious"] else "normal"
    typ = txn.get("suspicious_typology") or "-"
    print(
        f"[{label:>10}] {txn['account_id']} ({txn['customer_id']}) "
        f"| {txn['account_tier']} | NGN {txn['amount']:,.2f} | {typ}"
    )


def generate_population_stream(count):
    sent = 0
    weights = [w for _, _, w, _ in SUSPICIOUS_GENERATORS]
    generators = [(label, fn, is_seq) for label, fn, _, is_seq in SUSPICIOUS_GENERATORS]

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        print(f"Starting generation of {count} transactions...")
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
                else:
                    txn = fn()
                    _write_and_send(txn, f)
                    sent += 1
            else:
                txn = generate_normal_transaction()
                _write_and_send(txn, f)
                sent += 1

        print(f"Done. {sent} transactions written to {OUTPUT_FILE}")


if __name__ == "__main__":
    TARGET_COUNT = 5000
    print(f"TxGuard Simulator — generating {TARGET_COUNT} transactions")
    try:
        generate_population_stream(count=TARGET_COUNT)
    finally:
        close_producer()