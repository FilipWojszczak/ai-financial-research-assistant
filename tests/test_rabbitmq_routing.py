"""Opt-in broker checks using unique queues; never consume application jobs."""

import os
import time
import uuid
from contextlib import suppress
from unittest.mock import patch

import pytest
from amqp.exceptions import NotFound
from kombu import Connection, Exchange, Producer, Queue

from financial_assistant.core.config import get_settings
from financial_assistant.core.messaging import (
    declare_ingestion_topology,
    ingestion_queue,
)

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_RABBITMQ_TESTS") != "1",
    reason="Set RUN_RABBITMQ_TESTS=1 with an accessible test RabbitMQ broker",
)


@pytest.fixture
def broker_queues():
    prefix = "stage5_test_" + uuid.uuid4().hex
    failed_exchange = Exchange(prefix + "_failed", type="direct", durable=True)
    work_exchange = Exchange(prefix, type="topic", durable=True)
    failed = Queue(
        prefix + "_failed",
        exchange=failed_exchange,
        routing_key="failed",
        durable=True,
        queue_arguments={"x-queue-type": "quorum"},
    )
    work = Queue(
        prefix,
        exchange=work_exchange,
        routing_key="document.ingest",
        durable=True,
        queue_arguments={
            **ingestion_queue.queue_arguments,
            "x-dead-letter-exchange": failed_exchange.name,
        },
    )
    with Connection(
        get_settings().broker_url,
        connect_timeout=3,
        transport_options={
            "confirm_publish": True,
            "read_timeout": 3,
            "write_timeout": 3,
        },
    ) as connection:
        connection.ensure_connection(max_retries=0)
        try:
            with connection.channel() as channel:
                with (
                    patch(
                        "financial_assistant.core.messaging.dead_letter_queue", failed
                    ),
                    patch("financial_assistant.core.messaging.ingestion_queue", work),
                ):
                    declare_ingestion_topology(channel)
                producer = Producer(channel)
                producer.publish(
                    {"document_id": 42},
                    exchange=work_exchange,
                    routing_key=work.routing_key,
                    serializer="json",
                    delivery_mode=2,
                    retry=False,
                    confirm_timeout=3,
                )
                yield work(channel), failed(channel)
        finally:
            # Only these UUID-named test resources are removed, even after failure.
            for resource in (work, failed, work_exchange, failed_exchange):
                # A failed declaration may have closed its channel or created only
                # some resources. An absent resource must not prevent cleanup.
                with suppress(NotFound), connection.channel() as cleanup:
                    resource(cleanup).delete()


def get_message(queue):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        message = queue.get(accept=["json"])
        if message is not None:
            return message
        time.sleep(0.05)
    pytest.fail(f"No message arrived in {queue.name}")


def test_rejected_message_reaches_durable_failure_queue(broker_queues):
    work, failed = broker_queues
    get_message(work).reject(requeue=False)
    parked = get_message(failed)
    assert parked.payload == {"document_id": 42}
    assert parked.headers["x-death"][0]["reason"] == "rejected"
    parked.ack()
    assert work.get(accept=["json"]) is None


def test_repeated_crash_requeues_eventually_reach_failure_queue(broker_queues):
    work, failed = broker_queues
    deadline = time.monotonic() + 5
    deliveries = 0
    while time.monotonic() < deadline:
        parked = failed.get(accept=["json"])
        if parked is not None:
            assert parked.payload == {"document_id": 42}
            assert parked.headers["x-death"][0]["reason"] == "delivery_limit"
            assert deliveries <= 7
            parked.ack()
            return
        message = work.get(accept=["json"])
        if message is not None:
            deliveries += 1
            message.reject(requeue=True)
        time.sleep(0.05)
    pytest.fail("Repeated requeues were not dead-lettered within the delivery limit")
