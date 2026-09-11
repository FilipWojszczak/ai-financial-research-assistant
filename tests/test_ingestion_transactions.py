"""Exercise transaction and duplicate-delivery behavior against PostgreSQL."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from financial_assistant.ai.document_ingestion import (
    _mark_document_failed,
    ingest_document,
)
from financial_assistant.core.config import get_settings
from financial_assistant.models.document import (
    ChildChunk,
    Document,
    DocumentStatus,
    DocumentType,
    ParentChunk,
)

_MODULE = "financial_assistant.ai.document_ingestion"


@pytest.fixture
async def committed_document(session):
    # The existing 'session' fixture prepares tables; separate connections are required
    # to test real commits/rollbacks and PostgreSQL locks between competing attempts.
    engine = create_async_engine(get_settings().database_url, poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as setup:
        document = Document(
            filename="test1.pdf",
            company_ticker="TEST",
            document_type=DocumentType.ANNUAL_REPORT,
            year=2025,
            status=DocumentStatus.PROCESSING,
        )
        setup.add(document)
        await setup.commit()
        document_id = document.id
    try:
        yield factory, document_id
    finally:
        async with factory() as cleanup:
            await cleanup.execute(delete(Document).where(Document.id == document_id))
            await cleanup.commit()
        await engine.dispose()


@pytest.fixture
def fake_pipeline():
    parents = [{"chunk_index": 0, "content": "parent"}]
    children = [
        {
            "parent_index": 0,
            "chunk_index": 0,
            "content": "child",
            "embedding": [0.1] * 768,
        }
    ]
    with (
        patch(
            f"{_MODULE}.load_pdf_documents", new=AsyncMock(return_value=[object()])
        ) as load,
        patch(
            f"{_MODULE}.split_into_parent_and_child_chunks",
            return_value=(parents, children),
        ),
        patch(
            f"{_MODULE}.generate_child_embeddings", new=AsyncMock(return_value=children)
        ),
        patch(
            f"{_MODULE}.process_document_graph", new=AsyncMock(return_value=[])
        ) as graph,
        patch(f"{_MODULE}.process_document_communities", new_callable=AsyncMock),
    ):
        yield load, graph


async def assert_document_state(factory, document_id, status, chunks):
    async with factory() as check:
        document = await check.get(Document, document_id)
        assert document.status == status
        assert (
            await check.scalar(
                select(func.count())
                .select_from(ParentChunk)
                .where(ParentChunk.document_id == document_id)
            )
            == chunks
        )
        # The fake pipeline yields exactly one child for its one parent, so these
        # tests use one expected count for both tables. Real chunking may produce
        # different parent and child counts.
        assert (
            await check.scalar(
                select(func.count())
                .select_from(ChildChunk)
                .join(ParentChunk)
                .where(ParentChunk.document_id == document_id)
            )
            == chunks
        )


async def test_redelivery_after_commit_does_not_duplicate_data(
    committed_document, fake_pipeline
):
    factory, document_id = committed_document
    load, graph = fake_pipeline
    await ingest_document(document_id, session_factory=factory)
    await ingest_document(document_id, session_factory=factory)
    await assert_document_state(factory, document_id, DocumentStatus.COMPLETED, 1)
    assert load.await_count == graph.await_count == 1


async def test_failed_attempt_rolls_back_partial_chunks_then_retry_succeeds(
    committed_document, fake_pipeline
):
    factory, document_id = committed_document
    _, graph = fake_pipeline
    graph.side_effect = ConnectionError("graph service unavailable")
    with pytest.raises(ConnectionError):
        await ingest_document(document_id, session_factory=factory)
    await assert_document_state(factory, document_id, DocumentStatus.PROCESSING, 0)
    graph.side_effect = None
    await ingest_document(document_id, session_factory=factory)
    await assert_document_state(factory, document_id, DocumentStatus.COMPLETED, 1)


async def test_late_failure_does_not_overwrite_completed_document(
    committed_document, fake_pipeline
):
    factory, document_id = committed_document
    await ingest_document(document_id, session_factory=factory)
    await _mark_document_failed(document_id, session_factory=factory)
    await assert_document_state(factory, document_id, DocumentStatus.COMPLETED, 1)


async def test_terminal_failure_is_persisted_and_skipped_on_redelivery(
    committed_document, fake_pipeline
):
    factory, document_id = committed_document
    load, _ = fake_pipeline
    await _mark_document_failed(document_id, session_factory=factory)
    await ingest_document(document_id, session_factory=factory)
    await assert_document_state(factory, document_id, DocumentStatus.FAILED, 0)
    load.assert_not_awaited()


async def test_concurrent_deliveries_serialize_on_document_lock(
    committed_document, fake_pipeline
):
    factory, document_id = committed_document
    load, _ = fake_pipeline
    entered = asyncio.Event()
    release = asyncio.Event()

    async def pause_first_attempt(*args):
        entered.set()
        await release.wait()
        return [object()]

    load.side_effect = pause_first_attempt
    first = asyncio.create_task(ingest_document(document_id, session_factory=factory))
    second = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        second = asyncio.create_task(
            ingest_document(document_id, session_factory=factory)
        )
        # The second connection must not get past SELECT ... FOR UPDATE yet.
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(second), timeout=0.2)
        assert load.await_count == 1
        release.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=5)
    finally:
        release.set()
        for task in (first, second):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in (first, second) if task is not None),
            return_exceptions=True,
        )
    await assert_document_state(factory, document_id, DocumentStatus.COMPLETED, 1)
    assert load.await_count == 1
