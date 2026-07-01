import os
import json

from dotenv import load_dotenv
from azure.eventhub import EventHubProducerClient, EventData

load_dotenv()

CONNECTION_STR = os.getenv("EVENTHUB_CONNECTION_STR")
EVENTHUB_NAME = os.getenv("EVENTHUB_NAME")

_producer = None


def get_producer():
    global _producer
    if _producer is None:
        _producer = EventHubProducerClient.from_connection_string(
            conn_str=CONNECTION_STR, eventhub_name=EVENTHUB_NAME
        )
    return _producer


def send_transaction(txn: dict):
    producer = get_producer()
    batch = producer.create_batch()
    batch.add(EventData(json.dumps(txn)))
    producer.send_batch(batch)


def close_producer():
    global _producer
    if _producer is not None:
        _producer.close()
        _producer = None
