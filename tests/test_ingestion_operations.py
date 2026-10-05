import asyncio
import os
import time
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select

from financial_assistant import cleanup_storage, reconcile_ingestion
from financial_assistant.core.document_storage import lock_document_storage
from financial_assistant.core.messaging import INGESTION_TASK_NAME
from financial_assistant.models import Document, DocumentOutbox
from financial_assistant.models.document import DocumentStatus, DocumentType


async def create_document(factory, status=DocumentStatus.PROCESSING):
    async with factory() as session, session.begin():
        document = Document(
            filename="report.pdf",
            company_ticker="TEST",
            document_type=DocumentType.ANNUAL_REPORT,
            year=2026,
            status=status,
        )
        session.add(document)
        await session.flush()
        event = DocumentOutbox(document_id=document.id)
        session.add(event)
        await session.flush()
        return document.id, event.id


@pytest.fixture
def operations(private_database, tmp_path, monkeypatch):
    factory = private_database.factory
    monkeypatch.setattr(cleanup_storage, "async_session_maker", factory)
    monkeypatch.setattr(reconcile_ingestion, "async_session_maker", factory)
    monkeypatch.setattr(
        cleanup_storage,
        "get_settings",
        lambda: SimpleNamespace(document_storage_path=tmp_path),
    )
    return factory


@pytest.mark.parametrize("status", list(DocumentStatus))
async def test_reconciler_commits_only_processing_as_failed(operations, status):
    document_id, event_id = await create_document(operations, status)
    assert await reconcile_ingestion.mark_failed_task(event_id) is True
    async with operations() as session:
        document = await session.get(Document, document_id)
        expected = (
            DocumentStatus.FAILED if status == DocumentStatus.PROCESSING else status
        )
        assert document.status == expected
    # Repeated deliveries and a document removed by the user are harmless.
    assert await reconcile_ingestion.mark_failed_task(event_id) is True
    async with operations() as session, session.begin():
        document = await session.get(Document, document_id)
        await session.delete(document)
    assert await reconcile_ingestion.mark_failed_task(event_id) is True


async def test_reconciler_defers_to_an_active_worker(operations):
    document_id, event_id = await create_document(operations)
    async with operations() as worker, worker.begin():
        document = await worker.get(Document, document_id, with_for_update=True)
        assert await reconcile_ingestion.mark_failed_task(event_id) is False
        document.status = DocumentStatus.COMPLETED
    assert await reconcile_ingestion.mark_failed_task(event_id) is True
    async with operations() as session:
        assert (
            await session.get(Document, document_id)
        ).status == DocumentStatus.COMPLETED


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


def old_file(path):
    path.write_bytes(b"source")
    past = time.time() - 3600
    os.utime(path, (past, past))
    return path


async def test_cleanup_removes_only_old_unreferenced_files(operations, tmp_path):
    for status in DocumentStatus:
        document_id, _ = await create_document(operations, status)
        old_file(tmp_path / f"{document_id}.pdf")
    orphan = old_file(tmp_path / "999.pdf")
    abandoned = old_file(tmp_path / f".1.pdf.{uuid.uuid4().hex}.tmp")
    untouched = [
        old_file(tmp_path / "notes.pdf"),
        tmp_path / "1000.pdf",
        tmp_path / "1001.pdf",
    ]
    untouched[1].write_bytes(b"recent")
    untouched[2].symlink_to(untouched[0])
    assert await cleanup_storage.cleanup_orphaned_files(min_age_seconds=60) == 2
    assert not orphan.exists()
    assert not abandoned.exists()
    assert all(path.exists() for path in untouched)
    async with operations() as session:
        for document_id in await session.scalars(select(Document.id)):
            assert (tmp_path / f"{document_id}.pdf").exists()


async def test_cleanup_skips_even_old_files_of_uncommitted_uploads(
    operations, tmp_path
):
    async with operations() as upload:
        document = Document(
            filename="in-flight.pdf",
            company_ticker="TEST",
            document_type=DocumentType.ANNUAL_REPORT,
            year=2026,
        )
        upload.add(document)
        await upload.flush()
        await lock_document_storage(upload, document.id)
        source = old_file(tmp_path / f"{document.id}.pdf")
        temporary = old_file(tmp_path / f".{document.id}.pdf.{uuid.uuid4().hex}.tmp")
        assert await cleanup_storage.cleanup_orphaned_files(min_age_seconds=60) == 0
        assert source.exists() and temporary.exists()
        await upload.rollback()
    assert await cleanup_storage.cleanup_orphaned_files(min_age_seconds=60) == 2


async def test_database_failure_never_authorizes_file_deletion(operations, tmp_path):
    source = old_file(tmp_path / "999.pdf")
    failed_session = MagicMock()
    failed_session.__aenter__ = AsyncMock(side_effect=ConnectionError("database down"))
    with (
        patch.object(
            cleanup_storage, "async_session_maker", return_value=failed_session
        ),
        pytest.raises(ConnectionError),
    ):
        await cleanup_storage.cleanup_orphaned_files(min_age_seconds=60)
    assert source.exists()
