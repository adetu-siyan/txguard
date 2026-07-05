"""
TxGuard Simulator Producer — Phase 6
=======================================
Replaces eventhub_client.py. Sends transactions to the mock broker
instead of Azure Event Hubs. The interface is identical so main.py
doesn't need to change — just swap the import.

To switch to real Redpanda/Kafka later:
    Replace MockProducer with KafkaProducer from kafka-python:
    from kafka import KafkaProducer
    producer = KafkaProducer(
        bootstrap_servers="localhost:9092",
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )
    Everything else stays identical.
"""

import os
from pipeline.broker import MockProducer

TOPIC = "transactions"

_producer = None


def get_producer():
    global _producer
    if _producer is None:
        _producer = MockProducer()
    return _producer


def send_transaction(txn):
    """
    Sends one transaction dict to the transactions topic.
    Drop-in replacement for the Azure Event Hubs send_transaction().
    """
    get_producer().send(TOPIC, txn)


def close_producer():
    global _producer
    if _producer is not None:
        _producer.close()
        _producer = None
        