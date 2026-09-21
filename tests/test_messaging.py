import uuid
from unittest.mock import MagicMock, call, patch

import pytest

from financial_assistant.core.messaging import (
    INGESTION_QUEUE,
    INGESTION_TASK_NAME,
    declare_ingestion_topology,
)
from financial_assistant.core.outbox import publish_ingestion


def test_failure_destination_is_declared_before_work_queue():
    channel = MagicMock()
    calls = MagicMock()
    with (
        patch("financial_assistant.core.messaging.dead_letter_queue", new=calls.failed),
        patch("financial_assistant.core.messaging.ingestion_queue", new=calls.work),
    ):
        declare_ingestion_topology(channel)
    assert calls.mock_calls == [
        call.failed(channel),
        call.failed().declare(),
        call.work(channel),
        call.work().declare(),
    ]


def test_publisher_declares_topology_then_sends_persistent_id_only_message():
    event_id = uuid.uuid4()
    with (
        patch("financial_assistant.core.celery_app.celery_app") as app,
        patch("financial_assistant.core.outbox.declare_ingestion_topology") as declare,
    ):
        connection = app.connection_for_write.return_value.__enter__.return_value
        producer = app.amqp.Producer.return_value.__enter__.return_value
        calls = MagicMock()
        # Like patch's `new` argument, attach_mock reuses an existing mock; it
        # also parents that mock so calls across both patches share one history.
        calls.attach_mock(declare, "declare")
        calls.attach_mock(app.send_task, "send")
        publish_ingestion(42, event_id)

    connection.ensure_connection.assert_called_once_with(max_retries=0)
    app.connection_for_write.assert_called_once_with(
        connect_timeout=5,
        transport_options={
            "confirm_publish": True,
            "read_timeout": 5,
            "write_timeout": 5,
        },
    )
    assert calls.mock_calls == [
        call.declare(producer.channel),
        call.send(
            INGESTION_TASK_NAME,
            kwargs={"document_id": 42},
            task_id=str(event_id),
            queue=INGESTION_QUEUE,
            producer=producer,
            retry=False,
            delivery_mode=2,
            timeout=5,
            confirm_timeout=5,
        ),
    ]


@pytest.mark.parametrize("failure_at", ["declare", "publish"])
def test_broker_failures_propagate_so_the_outbox_remains_pending(failure_at):
    with (
        patch("financial_assistant.core.celery_app.celery_app") as app,
        patch("financial_assistant.core.outbox.declare_ingestion_topology") as declare,
    ):
        failing = declare if failure_at == "declare" else app.send_task
        failing.side_effect = ConnectionError("broker unavailable")
        with pytest.raises(ConnectionError):
            publish_ingestion(42, uuid.uuid4())
        if failure_at == "declare":
            app.send_task.assert_not_called()
