"""
TxGuard Mock Broker — Phase 6
================================
A lightweight Kafka-interface-compatible message broker backed by a
local queue and file. Mimics the producer/consumer API so all other
code is written against the Kafka interface — swap the connection
string later to connect to real Redpanda/Kafka without changing anything.

Two classes:
    MockProducer  — send(topic, message) — used by the simulator
    MockConsumer  — poll() — used by the pipeline consumer

Messages are stored in a JSONL file (data/broker_queue.jsonl) so they
persist across process restarts, simulating a real message queue's
durability guarantee.
"""

import os
import json
import time
import threading
from datetime import datetime
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUEUE_FILE = os.path.join(BASE_DIR, "data", "broker_queue.jsonl")
OFFSET_FILE = os.path.join(BASE_DIR, "data", "broker_offsets.json")

# Thread lock — prevents race conditions when simulator and consumer
# both access the queue file simultaneously
_file_lock = threading.Lock()


def _ensure_data_dir():
    os.makedirs(os.path.dirname(QUEUE_FILE), exist_ok=True)


def _load_offsets():
    """Loads consumer offsets — tracks which messages have been consumed."""
    if os.path.exists(OFFSET_FILE):
        with open(OFFSET_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_offsets(offsets):
    with open(OFFSET_FILE, "w") as f:
        json.dump(offsets, f)


class MockProducer:
    """
    Kafka-compatible producer interface.
    Writes messages to the local queue file.

    Usage (mirrors kafka-python KafkaProducer):
        producer = MockProducer()
        producer.send("transactions", transaction_dict)
        producer.flush()
        producer.close()
    """

    def __init__(self, bootstrap_servers=None):
        _ensure_data_dir()
        self.topic_counts = defaultdict(int)
        print(f"MockProducer initialized — queue: {QUEUE_FILE}")

    def send(self, topic, value):
        """
        Sends a message to the specified topic.
        value must be a dict — will be JSON serialized.
        """
        message = {
            "topic": topic,
            "value": value,
            "offset": self._get_next_offset(topic),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        with _file_lock:
            with open(QUEUE_FILE, "a") as f:
                f.write(json.dumps(message) + "\n")

        self.topic_counts[topic] += 1
        return self  # chainable

    def _get_next_offset(self, topic):
        offsets = _load_offsets()
        written_key = f"written_{topic}"
        current = offsets.get(written_key, 0)
        offsets[written_key] = current + 1
        _save_offsets(offsets)
        return current

    def flush(self):
        """No-op for mock — writes are immediate."""
        pass

    def close(self):
        total = sum(self.topic_counts.values())
        print(f"MockProducer closed — {total} messages sent across {len(self.topic_counts)} topics")
        for topic, count in self.topic_counts.items():
            print(f"  {topic}: {count} messages")


class MockConsumer:
    """
    Kafka-compatible consumer interface.
    Reads messages from the local queue file, tracking offset per topic
    so each message is consumed exactly once.

    Usage (mirrors kafka-python KafkaConsumer):
        consumer = MockConsumer("transactions")
        for message in consumer.poll(timeout_ms=1000):
            process(message.value)
        consumer.close()
    """

    def __init__(self, *topics, group_id="txguard-consumer", auto_offset_reset="earliest"):
        _ensure_data_dir()
        self.topics = list(topics)
        self.group_id = group_id
        self.auto_offset_reset = auto_offset_reset
        self._running = False
        print(f"MockConsumer initialized — topics: {self.topics}")

    def _get_consumed_offset(self, topic):
        offsets = _load_offsets()
        key = f"consumed_{self.group_id}_{topic}"
        if self.auto_offset_reset == "earliest":
            return offsets.get(key, 0)
        else:
            # latest — start from current end
            written_key = f"written_{topic}"
            offsets_data = _load_offsets()
            return offsets_data.get(written_key, 0)

    def _set_consumed_offset(self, topic, offset):
        offsets = _load_offsets()
        key = f"consumed_{self.group_id}_{topic}"
        offsets[key] = offset
        _save_offsets(offsets)

    def poll(self, timeout_ms=1000, max_records=100):
        """
        Reads up to max_records unconsumed messages from subscribed topics.
        Returns a list of message objects with .topic, .value, .offset attributes.
        Blocks for up to timeout_ms if no messages are available.
        """
        messages = []

        if not os.path.exists(QUEUE_FILE):
            time.sleep(timeout_ms / 1000)
            return messages

        # Build per-topic consumed offsets
        topic_offsets = {t: self._get_consumed_offset(t) for t in self.topics}
        topic_new_offsets = {t: topic_offsets[t] for t in self.topics}

        with _file_lock:
            with open(QUEUE_FILE, "r") as f:
                lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            topic = msg.get("topic")
            if topic not in self.topics:
                continue

            offset = msg.get("offset", 0)
            if offset < topic_offsets[topic]:
                continue  # already consumed

            # Build a message object
            message = _MockMessage(
                topic=topic,
                value=msg.get("value", {}),
                offset=offset,
                timestamp=msg.get("timestamp"),
            )
            messages.append(message)
            topic_new_offsets[topic] = max(topic_new_offsets[topic], offset + 1)

            if len(messages) >= max_records:
                break

        # Commit offsets for consumed messages
        for topic, new_offset in topic_new_offsets.items():
            if new_offset > topic_offsets[topic]:
                self._set_consumed_offset(topic, new_offset)

        if not messages:
            time.sleep(min(timeout_ms / 1000, 0.5))

        return messages

    def consume_forever(self, callback, poll_interval_ms=500):
        """
        Continuously polls for messages and calls callback(message) for each.
        Runs until stop() is called.

        callback signature: callback(message) where message has .topic and .value
        """
        self._running = True
        print(f"Consumer started — polling every {poll_interval_ms}ms")
        while self._running:
            messages = self.poll(timeout_ms=poll_interval_ms)
            for msg in messages:
                try:
                    callback(msg)
                except Exception as e:
                    print(f"  Error processing message: {e}")

    def stop(self):
        self._running = False

    def close(self):
        self._running = False
        print("MockConsumer closed.")

    def clear_queue(self):
        """Wipes the queue file and resets all offsets. Use for fresh runs."""
        if os.path.exists(QUEUE_FILE):
            os.remove(QUEUE_FILE)
        if os.path.exists(OFFSET_FILE):
            os.remove(OFFSET_FILE)
        print("Queue and offsets cleared.")


class _MockMessage:
    """Mimics kafka-python's ConsumerRecord namedtuple."""
    def __init__(self, topic, value, offset, timestamp=None):
        self.topic = topic
        self.value = value
        self.offset = offset
        self.timestamp = timestamp

    def __repr__(self):
        return (
            f"MockMessage(topic={self.topic}, "
            f"offset={self.offset}, "
            f"amount={self.value.get('amount', 'N/A')})"
        )


# ── Queue inspector ────────────────────────────────────────────────────────
def inspect_queue():
    """Prints a summary of messages currently in the queue."""
    if not os.path.exists(QUEUE_FILE):
        print("Queue is empty — no messages yet.")
        return

    topic_counts = defaultdict(int)
    total = 0

    with open(QUEUE_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                topic_counts[msg.get("topic", "unknown")] += 1
                total += 1
            except json.JSONDecodeError:
                continue

    offsets = _load_offsets()

    print(f"\nTxGuard Mock Broker Queue Status")
    print(f"  Queue file: {QUEUE_FILE}")
    print(f"  Total messages: {total}")
    print(f"\n  Per-topic breakdown:")
    for topic, count in topic_counts.items():
        written = offsets.get(f"written_{topic}", 0)
        consumed = offsets.get(f"consumed_txguard-consumer_{topic}", 0)
        pending = max(0, written - consumed)
        print(f"    {topic}: {count} total | {pending} pending consumption")


if __name__ == "__main__":
    inspect_queue()