import asyncio
import uuid
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import UploadFile
from sqlalchemy import delete, func, select, text
from sqlalchemy import event as sqlalchemy_event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from starlette.datastructures import Headers

from financial_assistant.api.routers.documents import upload_document
from financial_assistant.core.config import get_settings
from financial_assistant.core.outbox import publish_pending_documents
from financial_assistant.models import Base, Document, DocumentOutbox
from financial_assistant.models.document import DocumentStatus, DocumentType
from financial_assistant.schemas.document import DocumentCreate


@pytest.fixture
def send() -> Iterator[MagicMock]:
    with patch("financial_assistant.core.outbox.publish_ingestion") as mock_publish:
        yield mock_publish


@pytest.fixture
async def outbox_db() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    # Use a private schema so the dispatcher never claims unrelated test data.
    schema = "outbox_test_" + uuid.uuid4().hex
    engine = create_async_engine(
        get_settings().database_url,
        execution_options={"schema_translate_map": {None: schema}},
    )
    async with engine.begin() as conn:
        await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        with patch("financial_assistant.core.outbox.async_session_maker", factory):
            yield factory
    finally:
        async with engine.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await engine.dispose()


async def create_event(factory):
    async with factory() as session, session.begin():
        document = Document(
            filename="outbox-test.pdf",
            company_ticker="TEST",
            document_type=DocumentType.ANNUAL_REPORT,
            year=2026,
            status=DocumentStatus.PROCESSING,
        )
        session.add(document)
        await session.flush()
        event = DocumentOutbox(document_id=document.id)
        session.add(event)
        await session.flush()
        return event


async def read_event(factory, event_id):
    async with factory() as session:
        return await session.get(DocumentOutbox, event_id)


async def test_document_and_request_rollback_together(outbox_db):
    async with outbox_db() as session:
        document = Document(
            filename="rollback.pdf",
            company_ticker="TEST",
            document_type=DocumentType.ANNUAL_REPORT,
            year=2026,
        )
        session.add(document)
        await session.flush()
        document_id = document.id
        session.add(DocumentOutbox(document_id=document_id))
        await session.flush()
        await session.rollback()
    async with outbox_db() as check:
        assert await check.get(Document, document_id) is None
        assert await check.scalar(select(func.count()).select_from(DocumentOutbox)) == 0


async def test_success_is_recorded_once_and_not_republished(send, outbox_db):
    event = await create_event(outbox_db)
    assert await publish_pending_documents() == 1
    assert await publish_pending_documents() == 0
    send.assert_called_once_with(event.document_id, event.id)
    saved = await read_event(outbox_db, event.id)
    assert saved.published_at is not None
    assert saved.attempts == 1
    assert saved.last_error is None


async def test_broker_failure_remains_pending_then_recovers(send, outbox_db, caplog):
    event = await create_event(outbox_db)
    send.side_effect = ConnectionError("secret-broker-password")
    assert await publish_pending_documents() == 0
    saved = await read_event(outbox_db, event.id)
    assert saved.published_at is None
    assert saved.attempts == 1
    assert saved.last_error == "ConnectionError"
    assert saved.next_attempt_at > datetime.now(UTC)
    assert "secret-broker-password" not in caplog.text
    # The event is waiting to be retried (saved.next_attempt_at > datetime.now(UTC)),
    # so this pass does not call send.
    assert await publish_pending_documents() == 0
    assert send.call_count == 1
    async with outbox_db() as session, session.begin():
        row = await session.get(DocumentOutbox, event.id)
        row.next_attempt_at = datetime.now(UTC) - timedelta(seconds=1)
    send.side_effect = None
    assert await publish_pending_documents() == 1
    saved = await read_event(outbox_db, event.id)
    assert saved.attempts == 2
    assert saved.last_error is None


async def test_failed_event_does_not_block_later_events(send, outbox_db):
    first = await create_event(outbox_db)
    second = await create_event(outbox_db)

    def fail_first(document_id, event_id):
        if event_id == first.id:
            raise ConnectionError("broker down")

    send.side_effect = fail_first
    assert await publish_pending_documents() == 1
    assert (await read_event(outbox_db, first.id)).published_at is None
    assert (await read_event(outbox_db, second.id)).published_at is not None


async def test_locked_event_is_skipped_by_other_dispatcher(send, outbox_db):
    first = await create_event(outbox_db)
    second = await create_event(outbox_db)
    async with outbox_db() as owner, owner.begin():
        await owner.get(DocumentOutbox, first.id, with_for_update=True)
        count = await asyncio.wait_for(
            publish_pending_documents(),
            timeout=5,
        )
        assert count == 1
        send.assert_called_once_with(second.document_id, second.id)
    assert (await read_event(outbox_db, first.id)).published_at is None


async def test_commit_failure_after_publish_resends_same_task_id(send, outbox_db):
    event = await create_event(outbox_db)

    # Fail during transaction commit, after the broker call, without saving the
    # receipt. This models a process dying between publish and database commit.
    def fail_commit(session):
        raise RuntimeError("lost before commit")

    async with outbox_db() as failing:
        sqlalchemy_event.listen(failing.sync_session, "before_commit", fail_commit)
        with (
            patch(
                "financial_assistant.core.outbox.async_session_maker",
                return_value=failing,
            ),
            pytest.raises(RuntimeError, match="lost before commit"),
        ):
            await publish_pending_documents(limit=1)
    assert (await read_event(outbox_db, event.id)).published_at is None
    assert await publish_pending_documents() == 1
    assert send.call_count == 2
    assert send.call_args_list[0] == send.call_args_list[1]


async def test_deleting_document_cascades_to_pending_request(send, outbox_db):
    event = await create_event(outbox_db)
    async with outbox_db() as session, session.begin():
        await session.execute(delete(Document).where(Document.id == event.document_id))
    assert await read_event(outbox_db, event.id) is None
    assert await publish_pending_documents() == 0
    send.assert_not_called()


async def test_batch_limit_leaves_remaining_events_pending(send, outbox_db):
    await create_event(outbox_db)
    await create_event(outbox_db)
    assert await publish_pending_documents(limit=1) == 1
    assert send.call_count == 1
    assert await publish_pending_documents() == 1


@pytest.mark.parametrize("limit", [0, -1, True])
async def test_invalid_limit_is_rejected(limit):
    with pytest.raises(ValueError):
        await publish_pending_documents(limit=limit)


async def test_outbox_insert_failure_rolls_back_upload_and_cleans_source(outbox_db):
    def reject_insert(*args):
        raise RuntimeError("outbox insert failed")

    uploaded_file = UploadFile(
        file=BytesIO(b"pdf"),
        filename="report.pdf",
        headers=Headers({"content-type": "application/pdf"}),
    )
    sqlalchemy_event.listen(DocumentOutbox, "before_insert", reject_insert)
    try:
        async with outbox_db() as session:
            with (
                patch(
                    "financial_assistant.api.routers.documents.store_document_file",
                    new_callable=AsyncMock,
                ),
                patch(
                    "financial_assistant.api.routers.documents.delete_document_file",
                    new_callable=AsyncMock,
                ) as remove,
                pytest.raises(RuntimeError, match="outbox insert failed"),
            ):
                await upload_document(
                    document_data=DocumentCreate(
                        company_ticker="TEST",
                        document_type=DocumentType.ANNUAL_REPORT,
                        year=2026,
                        is_public=True,
                    ),
                    file=uploaded_file,
                    session=session,
                    user=MagicMock(id=1),
                )
        assert remove.await_count == 1
        async with outbox_db() as check:
            assert await check.scalar(select(func.count()).select_from(Document)) == 0
            assert (
                await check.scalar(select(func.count()).select_from(DocumentOutbox))
                == 0
            )
    finally:
        sqlalchemy_event.remove(DocumentOutbox, "before_insert", reject_insert)


async def test_uncommitted_request_is_invisible_to_publisher(send, outbox_db):
    async with outbox_db() as session, session.begin():
        document = Document(
            filename="pending.pdf",
            company_ticker="TEST",
            document_type=DocumentType.ANNUAL_REPORT,
            year=2026,
        )
        session.add(document)
        await session.flush()
        session.add(DocumentOutbox(document_id=document.id))
        await session.flush()
        assert await publish_pending_documents() == 0
        send.assert_not_called()
    assert await publish_pending_documents() == 1


async def test_cancellation_keeps_request_pending(send, outbox_db):
    event = await create_event(outbox_db)
    send.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await publish_pending_documents()
    saved = await read_event(outbox_db, event.id)
    assert saved.published_at is None
    assert saved.attempts == 0
    send.side_effect = None
    assert await publish_pending_documents() == 1


async def test_shutdown_finishes_current_row_and_leaves_rest_pending(send, outbox_db):
    first = await create_event(outbox_db)
    second = await create_event(outbox_db)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    def request_stop(document_id, event_id):
        # The broker publisher runs in a thread; signal the loop safely.
        loop.call_soon_threadsafe(stop.set)

    send.side_effect = request_stop
    assert await publish_pending_documents(stop=stop) == 1
    send.assert_called_once_with(first.document_id, first.id)
    assert (await read_event(outbox_db, first.id)).published_at is not None
    assert (await read_event(outbox_db, second.id)).published_at is None
