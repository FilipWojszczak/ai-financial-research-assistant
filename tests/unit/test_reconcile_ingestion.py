import asyncio
import uuid
from unittest.mock import MagicMock, patch

import pytest

from financial_assistant import reconcile_ingestion
from financial_assistant.core.messaging import INGESTION_TASK_NAME

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("outcome", [True, False, ConnectionError("database down")])
def test_failure_message_acknowledged_only_after_successful_reconciliation(outcome):
    events = []
    message = MagicMock(headers={"task": INGESTION_TASK_NAME, "id": str(uuid.uuid4())})
    message.ack.side_effect = lambda: events.append("ack")

    async def settle(task_id):
        events.append("database")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    with (
        patch.object(reconcile_ingestion.celery_app, "connection_for_read") as connect,
        patch.object(reconcile_ingestion, "dead_letter_queue") as queue,
        patch.object(reconcile_ingestion, "mark_failed_task", side_effect=settle),
        asyncio.Runner() as runner,
    ):
        connection = connect.return_value.__enter__.return_value
        queue.return_value.get.side_effect = [message, None]
        if isinstance(outcome, Exception):
            with pytest.raises(ConnectionError):
                reconcile_ingestion.reconcile_once(runner)
        else:
            assert reconcile_ingestion.reconcile_once(runner) == int(outcome)
        connection.channel.return_value.close.assert_called_once()
    assert events == (["database", "ack"] if outcome is True else ["database"])
