import asyncio
from typing import Protocol, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from celery.exceptions import Reject, Retry
from celery.result import EagerResult

from financial_assistant.ai.document_ingestion import (
    InvalidDocumentError,
    _mark_document_failed,
    ingest_document,
    load_pdf_documents,
)
from financial_assistant.core.celery_app import celery_app
from financial_assistant.models.document import Document, DocumentStatus
from financial_assistant.tasks.async_runner import (
    close_worker_runner,
    get_worker_runner,
)
from financial_assistant.tasks.document_ingestion import (
    _run_attempt,
    ingest_document_task,
)

_TASK_MODULE = "financial_assistant.tasks.document_ingestion"
_PIPELINE_MODULE = "financial_assistant.ai.document_ingestion"


class EagerCeleryTask(Protocol):
    """Celery task proxy API exercised by this module's tests."""

    name: str
    acks_late: bool
    reject_on_worker_lost: bool
    acks_on_failure_or_timeout: bool

    def _get_current_object(self) -> object: ...

    def apply(
        self,
        args: tuple[object, ...] | None = None,
        *,
        throw: bool = False,
        retries: int | None = None,
    ) -> EagerResult: ...


# Celery returns a runtime task proxy; its task attributes are not inferred.
ingest_task = cast(EagerCeleryTask, ingest_document_task)


@pytest.fixture(autouse=True)
def close_task_loop():
    yield
    close_worker_runner()


def test_task_is_registered_with_delivery_guarantees():
    celery_app.loader.import_default_modules()
    assert celery_app.tasks[ingest_task.name] is ingest_task._get_current_object()
    assert ingest_task.acks_late is True
    assert ingest_task.reject_on_worker_lost is True
    assert ingest_task.acks_on_failure_or_timeout is True
    assert celery_app.conf.worker_prefetch_multiplier == 1


def test_successful_task_passes_only_document_id_to_attempt():
    with patch(f"{_TASK_MODULE}._run_attempt", new_callable=AsyncMock) as attempt:
        result = ingest_task.apply(args=(42,), throw=True)
    assert result.successful()
    attempt.assert_awaited_once_with(42, final_attempt=False)


@pytest.mark.parametrize(("retries", "delay"), [(0, 10), (1, 20), (2, 40)])
def test_retryable_errors_schedule_bounded_backoff(retries, delay):
    error = ConnectionError("provider unavailable")
    with (
        patch(f"{_TASK_MODULE}._run_attempt", new=AsyncMock(side_effect=error)),
        pytest.raises(Retry) as retry,
    ):
        ingest_task.apply(args=(42,), retries=retries, throw=True)
    assert retry.value.when == delay
    assert retry.value.exc is error


def test_persistent_error_stops_after_four_attempts():
    attempt = AsyncMock(side_effect=ConnectionError("provider unavailable"))
    with patch(f"{_TASK_MODULE}._run_attempt", new=attempt):
        # Celery eager execution follows retries synchronously, without RabbitMQ.
        result = ingest_task.apply(args=(42,), throw=False)
    assert result.state == "REJECTED"
    assert isinstance(result.result, Reject)
    assert result.result.requeue is False
    assert attempt.await_count == 4
    assert [call.kwargs["final_attempt"] for call in attempt.await_args_list] == [
        False,
        False,
        False,
        True,
    ]


def test_invalid_pdf_is_not_retried():
    attempt = AsyncMock(side_effect=InvalidDocumentError("empty PDF"))
    with patch(f"{_TASK_MODULE}._run_attempt", new=attempt):
        result = ingest_task.apply(args=(42,), throw=True)
    assert result.state == "REJECTED"
    assert isinstance(result.result, Reject)
    assert result.result.requeue is False
    assert attempt.await_count == 1


@pytest.mark.parametrize("document_id", [0, -1, "42", True, None])
def test_invalid_message_is_rejected_before_opening_resources(document_id):
    with (
        patch(f"{_TASK_MODULE}._run_attempt", new_callable=AsyncMock) as attempt,
    ):
        result = ingest_task.apply(args=(document_id,), throw=True)
    assert result.state == "REJECTED"
    assert isinstance(result.result, Reject)
    assert result.result.requeue is False
    assert "positive integer" in str(result.result)
    attempt.assert_not_called()


def test_sequential_tasks_share_worker_event_loop():
    loops = []

    async def attempt(*args, **kwargs):
        loops.append(asyncio.get_running_loop())

    with patch(f"{_TASK_MODULE}._run_attempt", side_effect=attempt):
        ingest_task.apply(args=(42,), throw=True)
        ingest_task.apply(args=(43,), throw=True)
    assert loops[0] is loops[1]
    close_worker_runner()
    assert loops[0].is_closed()
    assert get_worker_runner().get_loop() is not loops[0]


@pytest.mark.parametrize(
    ("error", "final_attempt", "marks_failed"),
    [
        (ConnectionError("unavailable"), False, False),
        (ConnectionError("unavailable"), True, True),
        (InvalidDocumentError("invalid PDF"), False, True),
        (asyncio.CancelledError(), False, False),
    ],
)
async def test_attempt_marks_only_terminal_failure_and_closes_engine(
    error, final_attempt, marks_failed
):
    engine = MagicMock(dispose=AsyncMock())
    factory = MagicMock()
    with (
        patch(f"{_TASK_MODULE}.create_async_engine", return_value=engine),
        patch(f"{_TASK_MODULE}.async_sessionmaker", return_value=factory),
        patch(f"{_TASK_MODULE}.ingest_document", new=AsyncMock(side_effect=error)),
        patch(f"{_TASK_MODULE}._mark_document_failed", new_callable=AsyncMock) as mark,
        pytest.raises(type(error)),
    ):
        await _run_attempt(42, final_attempt=final_attempt)
    engine.dispose.assert_awaited_once_with()
    if marks_failed:
        mark.assert_awaited_once_with(42, session_factory=factory)
    else:
        mark.assert_not_awaited()


async def test_successful_attempt_disposes_engine_without_marking_failed():
    engine = MagicMock(dispose=AsyncMock())
    factory = MagicMock()
    with (
        patch(f"{_TASK_MODULE}.create_async_engine", return_value=engine),
        patch(f"{_TASK_MODULE}.async_sessionmaker", return_value=factory),
        patch(f"{_TASK_MODULE}.ingest_document", new_callable=AsyncMock) as ingest,
        patch(f"{_TASK_MODULE}._mark_document_failed", new_callable=AsyncMock) as mark,
    ):
        await _run_attempt(42, final_attempt=False)
    ingest.assert_awaited_once_with(42, session_factory=factory)
    engine.dispose.assert_awaited_once_with()
    mark.assert_not_awaited()


@pytest.mark.parametrize(
    "status", [None, DocumentStatus.COMPLETED, DocumentStatus.FAILED]
)
async def test_terminal_or_deleted_document_skips_all_processing(status):
    document = None if status is None else MagicMock(status=status)
    session = MagicMock(get=AsyncMock(return_value=document), rollback=AsyncMock())
    factory = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(return_value=session)
    factory.return_value.__aexit__ = AsyncMock(return_value=False)
    with patch(
        f"{_PIPELINE_MODULE}.load_pdf_documents", new_callable=AsyncMock
    ) as load:
        await ingest_document(42, session_factory=factory)
    session.get.assert_awaited_once_with(Document, 42, with_for_update=True)
    load.assert_not_awaited()
    session.add_all.assert_not_called()


async def test_missing_source_is_permanent(tmp_path):
    with pytest.raises(InvalidDocumentError):
        await load_pdf_documents(tmp_path / "missing.pdf")


async def test_corrupt_source_is_permanent(tmp_path):
    corrupt = tmp_path / "corrupt.pdf"
    # These bytes are plain text, not the structured data required by a PDF.
    corrupt.write_bytes(b"not a PDF")
    with pytest.raises(InvalidDocumentError):
        await load_pdf_documents(corrupt)


async def test_failure_status_write_error_is_logged(caplog):
    factory = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(side_effect=ConnectionError("db down"))
    await _mark_document_failed(42, session_factory=factory)
    assert "Failed to update status to FAILED for document 42" in caplog.text
