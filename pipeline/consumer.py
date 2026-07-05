"""
TxGuard Pipeline Consumer — Phase 6
======================================
Reads transactions from the mock broker, scores them through the
unified scorer, writes results to PostgreSQL.

Flow:
    MockBroker (transactions topic)
        → TxGuardScorer (ISO + RF + GNN)
        → PostgreSQL (transactions table + alerts table + customer_snapshots)

To switch to real Kafka/Redpanda later:
    Replace MockConsumer with KafkaConsumer from kafka-python.
    Everything else stays identical.
"""

import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.broker import MockConsumer
from pipeline.database import get_session, init_db, Transaction, Alert, CustomerSnapshot
from scorer.unified import TxGuardScorer

# ── Configuration ─────────────────────────────────────────────────────────
TOPIC = "transactions"
ALERT_THRESHOLD = 40.0   # minimum score to write an alert (MEDIUM+)
POLL_INTERVAL_MS = 500
BATCH_SIZE = 50           # commit to DB every N transactions


def parse_timestamp(ts_str):
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return datetime.utcnow()


def upsert_customer_snapshot(session, alert, txn):
    """
    Updates or creates the customer behavioral snapshot.
    One row per customer — updated after every transaction.
    """
    customer_id = txn.get("customer_id")
    if not customer_id:
        return

    existing = session.query(CustomerSnapshot).filter_by(
        customer_id=customer_id
    ).first()

    amount = txn.get("amount", 0)
    ts = parse_timestamp(txn.get("timestamp", ""))

    if existing:
        # Update existing snapshot
        existing.total_transactions += 1
        existing.total_amount += amount
        existing.mean_amount = existing.total_amount / existing.total_transactions
        existing.max_amount = max(existing.max_amount or 0, amount)

        if txn.get("is_suspicious"):
            existing.suspicious_count += 1

        if alert and alert["txguard_risk_score"] >= ALERT_THRESHOLD:
            existing.alert_count += 1
            existing.latest_risk_score = alert["txguard_risk_score"]
            existing.latest_risk_tier = alert["risk_tier"]
            existing.highest_risk_score = max(
                existing.highest_risk_score or 0,
                alert["txguard_risk_score"]
            )

        # Update account list
        current_accounts = existing.account_ids or []
        account_id = txn.get("account_id")
        if account_id and account_id not in current_accounts:
            current_accounts.append(account_id)
            existing.account_ids = current_accounts
            existing.account_count = len(current_accounts)

        existing.last_seen = ts
        existing.snapshot_updated_at = datetime.utcnow()
    else:
        # Create new snapshot using raw SQL upsert to handle
        # concurrent inserts within the same batch gracefully
        from sqlalchemy import text
        session.execute(text("""
            INSERT INTO customer_snapshots (
                customer_id, account_ids, account_count, account_tier,
                total_transactions, total_amount, mean_amount, max_amount,
                suspicious_count, alert_count, latest_risk_score,
                highest_risk_score, latest_risk_tier, first_seen,
                last_seen, snapshot_updated_at
            ) VALUES (
                :customer_id, :account_ids, :account_count, :account_tier,
                :total_transactions, :total_amount, :mean_amount, :max_amount,
                :suspicious_count, :alert_count, :latest_risk_score,
                :highest_risk_score, :latest_risk_tier, :first_seen,
                :last_seen, :snapshot_updated_at
            )
            ON CONFLICT (customer_id) DO UPDATE SET
                total_transactions = customer_snapshots.total_transactions + 1,
                total_amount = customer_snapshots.total_amount + :total_amount,
                mean_amount = (customer_snapshots.total_amount + :total_amount) /
                              (customer_snapshots.total_transactions + 1),
                max_amount = GREATEST(customer_snapshots.max_amount, :max_amount),
                suspicious_count = customer_snapshots.suspicious_count + :suspicious_count,
                alert_count = customer_snapshots.alert_count + :alert_count,
                latest_risk_score = :latest_risk_score,
                highest_risk_score = GREATEST(
                    COALESCE(customer_snapshots.highest_risk_score, 0),
                    COALESCE(:highest_risk_score, 0)
                ),
                latest_risk_tier = :latest_risk_tier,
                last_seen = :last_seen,
                snapshot_updated_at = :snapshot_updated_at
        """), {
            "customer_id": customer_id,
            "account_ids": json.dumps([txn.get("account_id")]),
            "account_count": 1,
            "account_tier": txn.get("account_tier"),
            "total_transactions": 1,
            "total_amount": amount,
            "mean_amount": amount,
            "max_amount": amount,
            "suspicious_count": 1 if txn.get("is_suspicious") else 0,
            "alert_count": 1 if (alert and alert["txguard_risk_score"] >= ALERT_THRESHOLD) else 0,
            "latest_risk_score": alert["txguard_risk_score"] if alert else None,
            "highest_risk_score": alert["txguard_risk_score"] if alert else None,
            "latest_risk_tier": alert["risk_tier"] if alert else None,
            "first_seen": ts,
            "last_seen": ts,
            "snapshot_updated_at": datetime.utcnow(),
        })

def process_message(msg, scorer, session, stats):
    """
    Processes one transaction message:
    1. Write raw transaction to transactions table
    2. Score through unified scorer
    3. Write alert to alerts table if score >= threshold
    4. Update customer snapshot
    """
    txn = msg.value

    # ── Write raw transaction ─────────────────────────────────────────────
    try:
        db_txn = Transaction(
            transaction_id=txn.get("transaction_id", f"TXN-{datetime.utcnow().timestamp()}"),
            account_id=txn.get("account_id", "UNKNOWN"),
            customer_id=txn.get("customer_id", "UNKNOWN"),
            account_tier=txn.get("account_tier", "TIER_2"),
            amount=txn.get("amount", 0),
            currency=txn.get("currency", "NGN"),
            txn_type=txn.get("type", "transfer"),
            channel=txn.get("channel", "mobile"),
            location=txn.get("location"),
            balance_after=txn.get("balance_after"),
            txn_timestamp=parse_timestamp(txn.get("timestamp", "")),
            is_suspicious=bool(txn.get("is_suspicious", False)),
            suspicious_typology=txn.get("suspicious_typology"),
        )
        session.merge(db_txn)  # merge = upsert by primary key
        stats["transactions"] += 1
    except Exception as e:
        print(f"  Error writing transaction: {e}")
        return

    # ── Score transaction ─────────────────────────────────────────────────
    try:
        alert = scorer.score(txn)
    except Exception as e:
        print(f"  Error scoring transaction: {e}")
        alert = None

    # ── Write alert if threshold met ──────────────────────────────────────
    if alert and alert["txguard_risk_score"] >= ALERT_THRESHOLD:
        try:
            db_alert = Alert(
                transaction_id=alert.get("transaction_id", "UNKNOWN"),
                account_id=alert.get("account_id", "UNKNOWN"),
                customer_id=alert.get("customer_id", "UNKNOWN"),
                account_tier=alert.get("account_tier"),
                amount=alert.get("amount"),
                txn_timestamp=parse_timestamp(alert.get("timestamp", "")),
                iso_score=alert.get("iso_score"),
                rf_score=alert.get("rf_score"),
                gnn_score=alert.get("gnn_score"),
                txguard_risk_score=alert.get("txguard_risk_score"),
                risk_tier=alert.get("risk_tier"),
                triggered_rules=alert.get("triggered_rules", []),
                regulatory_reference=alert.get("regulatory_reference"),
                legal_consequence=alert.get("legal_consequence"),
                recommended_action=alert.get("recommended_action"),
                scored_at=parse_timestamp(alert.get("scored_at", "")),
            )
            session.add(db_alert)
            stats["alerts"] += 1

            if alert["risk_tier"] == "HIGH":
                stats["high_alerts"] += 1
                print(
                    f"  [HIGH] {alert['account_id']} ({alert['customer_id']}) "
                    f"Score: {alert['txguard_risk_score']}/100 | "
                    f"₦{alert['amount']:,.0f} | "
                    f"{', '.join(alert['triggered_rules']) or 'ML flag'}"
                )
        except Exception as e:
            print(f"  Error writing alert: {e}")

    # ── Update customer snapshot ──────────────────────────────────────────
    try:
        upsert_customer_snapshot(session, alert, txn)
    except Exception as e:
        print(f"  Error updating snapshot: {e}")


def run_consumer(max_messages=None, batch_size=BATCH_SIZE):
    """
    Runs the pipeline consumer.
    Reads from broker → scores → writes to PostgreSQL.

    Args:
        max_messages: stop after this many messages (None = run forever)
        batch_size: commit to DB every N transactions
    """
    print("Initializing TxGuard pipeline consumer...")
    init_db()

    print("Loading scorer...")
    scorer = TxGuardScorer()

    consumer = MockConsumer(TOPIC, group_id="txguard-consumer")
    session = get_session()

    stats = {
        "transactions": 0,
        "alerts": 0,
        "high_alerts": 0,
        "errors": 0,
    }

    print(f"\nConsuming from topic '{TOPIC}'...")
    print(f"Alert threshold: {ALERT_THRESHOLD}/100")
    print(f"Batch size: {batch_size}")
    print("=" * 60)

    try:
        while True:
            messages = consumer.poll(timeout_ms=1000, max_records=batch_size)

            if not messages:
                if max_messages and stats["transactions"] >= max_messages:
                    break
                if max_messages is None:
                    continue
                break

            for msg in messages:
                try:
                    process_message(msg, scorer, session, stats)
                except Exception as e:
                    stats["errors"] += 1
                    print(f"  Unhandled error: {e}")

            # Batch commit
            try:
                session.commit()
            except Exception as e:
                print(f"  Commit error: {e}")
                session.rollback()

            if max_messages and stats["transactions"] >= max_messages:
                break

    except KeyboardInterrupt:
        print("\nConsumer interrupted.")
    finally:
        session.commit()
        session.close()
        consumer.close()

    print(f"\n{'='*60}")
    print(f"  Consumer finished")
    print(f"  Transactions processed: {stats['transactions']}")
    print(f"  Alerts written:         {stats['alerts']}")
    print(f"  HIGH alerts:            {stats['high_alerts']}")
    print(f"  Errors:                 {stats['errors']}")
    print(f"{'='*60}")

    return stats


if __name__ == "__main__":
    # Process first 500 messages from the broker for a quick demo run
    run_consumer(max_messages=500)
