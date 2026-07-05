"""
TxGuard Database Schema — Phase 6
===================================
PostgreSQL tables for the TxGuard pipeline.
All writes from the unified scorer and investigator land here.
All reads from the FastAPI endpoint come from here.

Tables:
    transactions    — raw transaction stream (every event that flows through)
    alerts          — scored alerts from the unified scorer
    investigations  — STR drafts and investigation reports from the agent
    customer_state  — per-customer behavioral snapshot (updated incrementally)

Usage:
    from pipeline.database import init_db, get_session
    init_db()  # creates tables if they don't exist
    session = get_session()
"""

import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Float,
    Integer,
    Boolean,
    DateTime,
    Text,
    JSON,
    Index,
)
from sqlalchemy.orm import declarative_base, sessionmaker

# ── Connection ─────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL not found in environment.\n"
        "Add to .env: DATABASE_URL=postgresql://txguard_user:txguard2026@localhost:5432/txguard"
    )

engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,  # verify connection before using from pool
    echo=False,          # set True to log all SQL queries for debugging
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


# ── Table 1: Transactions ──────────────────────────────────────────────────
# Every transaction that flows through the pipeline is stored here.
# This is the raw event log — the source of truth for all downstream analysis.
# The investigator and scorer both read from this table when building context.

class Transaction(Base):
    __tablename__ = "transactions"

    # Primary key
    transaction_id = Column(String(50), primary_key=True)

    # Account and customer identity
    account_id     = Column(String(20), nullable=False, index=True)
    customer_id    = Column(String(20), nullable=False, index=True)
    account_tier   = Column(String(10), nullable=False)

    # Transaction details
    amount         = Column(Float, nullable=False)
    currency       = Column(String(5), default="NGN")
    txn_type       = Column(String(20), nullable=False)
    channel        = Column(String(20), nullable=False)
    location       = Column(String(50))
    balance_after  = Column(Float)

    # Timing
    txn_timestamp  = Column(DateTime, nullable=False, index=True)
    ingested_at    = Column(DateTime, default=datetime.utcnow)

    # Ground truth label (synthetic only — never used in detection)
    is_suspicious  = Column(Boolean, default=False)
    suspicious_typology = Column(String(50))

    __table_args__ = (
        # Composite index for the most common query pattern:
        # "give me all transactions for this customer in the last N hours"
        Index("ix_transactions_customer_time", "customer_id", "txn_timestamp"),
        Index("ix_transactions_account_time", "account_id", "txn_timestamp"),
    )


# ── Table 2: Alerts ────────────────────────────────────────────────────────
# One row per scored transaction that exceeded the risk threshold.
# Written by the unified scorer whenever txguard_risk_score >= 40 (MEDIUM+).
# The FastAPI /alerts endpoint reads from this table.

class Alert(Base):
    __tablename__ = "alerts"

    id             = Column(Integer, primary_key=True, autoincrement=True)

    # Transaction reference
    transaction_id = Column(String(50), nullable=False, index=True)
    account_id     = Column(String(20), nullable=False, index=True)
    customer_id    = Column(String(20), nullable=False, index=True)
    account_tier   = Column(String(10))
    amount         = Column(Float)
    txn_timestamp  = Column(DateTime)

    # Model scores
    iso_score      = Column(Float)
    rf_score       = Column(Float)
    gnn_score      = Column(Float)

    # Fused output
    txguard_risk_score = Column(Float, nullable=False, index=True)
    risk_tier      = Column(String(10), nullable=False, index=True)

    # Rule engine output
    triggered_rules = Column(JSON)  # list of rule names

    # Regulatory output
    regulatory_reference = Column(Text)
    legal_consequence    = Column(Text)
    recommended_action   = Column(Text)

    # Alert lifecycle
    scored_at      = Column(DateTime, default=datetime.utcnow)
    reviewed       = Column(Boolean, default=False)
    mlro_decision  = Column(String(20), default="PENDING")
    # PENDING / FILE_STR / ENHANCED_MONITORING / DISMISSED

    __table_args__ = (
        Index("ix_alerts_risk_tier_scored", "risk_tier", "scored_at"),
        Index("ix_alerts_customer_scored", "customer_id", "scored_at"),
    )


# ── Table 3: Investigations ────────────────────────────────────────────────
# One row per investigation run by the TxGuardInvestigator agent.
# Linked to an alert by alert_id. Stores the full investigation report
# including the STR draft for MLRO review.

class Investigation(Base):
    __tablename__ = "investigations"

    id              = Column(Integer, primary_key=True, autoincrement=True)

    # Alert reference
    alert_id        = Column(Integer, nullable=False, index=True)
    transaction_id  = Column(String(50), nullable=False)
    account_id      = Column(String(20), nullable=False, index=True)
    customer_id     = Column(String(20), nullable=False, index=True)
    all_accounts    = Column(JSON)  # list of all account_ids for this customer

    # Investigation output
    pattern_analysis      = Column(Text)
    cross_account_analysis = Column(Text)
    risk_assessment       = Column(Text)
    str_draft             = Column(Text)

    # Recommendation and decision
    final_recommendation  = Column(String(30), default="PENDING")
    # PENDING / FILE_STR / ENHANCED_MONITORING / DISMISS
    mlro_decision         = Column(String(30), default="PENDING")
    mlro_notes            = Column(Text)
    mlro_decided_at       = Column(DateTime)

    # Metadata
    model_used      = Column(String(50))
    investigated_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_investigations_customer", "customer_id"),
        Index("ix_investigations_status", "final_recommendation", "mlro_decision"),
    )


# ── Table 4: Customer State Snapshots ─────────────────────────────────────
# Periodic snapshots of each customer's behavioral state.
# Written by the pipeline consumer after processing each transaction.
# Used by the investigator to quickly pull "what does this customer look like"
# without replaying the entire transaction history.
# One row per customer — upserted (insert or update) on each transaction.

class CustomerSnapshot(Base):
    __tablename__ = "customer_snapshots"

    customer_id         = Column(String(20), primary_key=True)

    # Account summary
    account_ids         = Column(JSON)   # list of account_ids
    account_count       = Column(Integer, default=1)
    account_tier        = Column(String(10))

    # Behavioral summary (updated incrementally)
    total_transactions  = Column(Integer, default=0)
    total_amount        = Column(Float, default=0.0)
    mean_amount         = Column(Float, default=0.0)
    max_amount          = Column(Float, default=0.0)
    suspicious_count    = Column(Integer, default=0)
    alert_count         = Column(Integer, default=0)

    # Risk summary
    latest_risk_score   = Column(Float)
    highest_risk_score  = Column(Float)
    latest_risk_tier    = Column(String(10))

    # Timestamps
    first_seen          = Column(DateTime)
    last_seen           = Column(DateTime)
    snapshot_updated_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_snapshots_risk", "latest_risk_tier", "highest_risk_score"),
    )


# ── Helpers ────────────────────────────────────────────────────────────────
def init_db():
    """
    Creates all tables if they don't exist.
    Safe to call multiple times — won't drop or modify existing tables.
    Call this once at pipeline startup.
    """
    Base.metadata.create_all(bind=engine)
    print("TxGuard database initialized.")
    print(f"  Tables: {', '.join(Base.metadata.tables.keys())}")


def get_session():
    """
    Returns a new database session.
    Always close the session after use:
        session = get_session()
        try:
            # do work
            session.commit()
        except Exception:
            session.rollback()
        finally:
            session.close()
    """
    return SessionLocal()


def drop_all():
    """
    Drops all TxGuard tables. Destructive — use only during development
    when you need a clean slate.
    """
    Base.metadata.drop_all(bind=engine)
    print("All TxGuard tables dropped.")


# ── Test connection ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing database connection...")
    try:
        with engine.connect() as conn:
            print("Connection successful.")
        init_db()
        print("Schema created successfully.")
        print("\nTable summary:")
        for table_name in Base.metadata.tables:
            table = Base.metadata.tables[table_name]
            print(f"  {table_name}: {len(table.columns)} columns")
    except Exception as e:
        print(f"Error: {e}")