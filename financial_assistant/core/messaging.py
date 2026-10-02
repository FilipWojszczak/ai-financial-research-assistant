"""Shared RabbitMQ routing for document ingestion and parked failures."""

import re
from typing import Any

from kombu import Exchange, Queue

INGESTION_TASK_NAME = "financial_assistant.tasks.document_ingestion.ingest_document"
INGESTION_EXCHANGE = "document_tasks"
INGESTION_QUEUE = "document_ingestion"
INGESTION_ROUTING_KEY = "document.ingest"
DEAD_LETTER_QUEUE = "document_ingestion_failed"
DEAD_LETTER_EXCHANGE = "document_ingestion_dead_letters"
DEAD_LETTER_ROUTING_KEY = "failed"

INGESTION_QUEUE_ARGUMENTS: dict[str, str | int] = {
    "x-queue-type": "quorum",
    "x-delivery-limit": 5,
    "x-dead-letter-exchange": DEAD_LETTER_EXCHANGE,
    "x-dead-letter-routing-key": DEAD_LETTER_ROUTING_KEY,
    "x-dead-letter-strategy": "at-least-once",
    "x-overflow": "reject-publish",
}

INGESTION_POLICY_NAME = "document-ingestion"


def ingestion_policy(consumer_timeout_seconds: int) -> dict[str, Any]:
    """
    RabbitMQ policy for the ingestion queue. A policy (not a queue argument) is used
    because changing x-arguments of an existing queue fails its redeclaration.
    Only one policy applies to a queue, so this one must carry every key it needs.
    """
    return {
        "pattern": f"^{re.escape(INGESTION_QUEUE)}$",
        "apply-to": "queues",
        "priority": 10,
        "definition": {"consumer-timeout": consumer_timeout_seconds * 1000},
    }


dead_letter_queue = Queue(
    DEAD_LETTER_QUEUE,
    exchange=Exchange(DEAD_LETTER_EXCHANGE, type="direct", durable=True),
    routing_key=DEAD_LETTER_ROUTING_KEY,
    durable=True,
    # Keep failures available even after repeated consumer crashes.
    queue_arguments={"x-queue-type": "quorum", "x-delivery-limit": -1},
)

ingestion_queue = Queue(
    INGESTION_QUEUE,
    # Celery's native delayed delivery (used by retries) requires a topic exchange.
    exchange=Exchange(INGESTION_EXCHANGE, type="topic", durable=True),
    routing_key=INGESTION_ROUTING_KEY,
    durable=True,
    queue_arguments=INGESTION_QUEUE_ARGUMENTS.copy(),
)


def declare_ingestion_topology(channel) -> None:
    """Declare the failure destination before any ingestion can be delivered."""
    dead_letter_queue(channel).declare()
    ingestion_queue(channel).declare()
