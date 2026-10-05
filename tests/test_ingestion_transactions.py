"""Exercise transaction and duplicate-delivery behavior against PostgreSQL."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from financial_assistant.ai.checkpoints import AttemptCheckpoints
from financial_assistant.ai.community_detection import CommunitySummary
from financial_assistant.ai.document_ingestion import (
    DocumentBusyError,
    _mark_document_failed,
    ingest_document,
)
from financial_assistant.ai.graph_extraction import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)
from financial_assistant.core.config import get_settings
from financial_assistant.models.checkpoint import CheckpointKind, IngestionCheckpoint
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


async def test_duplicate_delivery_does_not_wait_for_active_attempt(
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
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        # The duplicate fails fast instead of holding a worker until the first ends.
        with pytest.raises(DocumentBusyError):
            await asyncio.wait_for(
                ingest_document(document_id, session_factory=factory), timeout=2
            )
        assert load.await_count == 1
        release.set()
        await asyncio.wait_for(first, timeout=5)
    finally:
        release.set()
        if not first.done():
            first.cancel()
        await asyncio.gather(first, return_exceptions=True)
    await assert_document_state(factory, document_id, DocumentStatus.COMPLETED, 1)

    # The re-queued duplicate later finds the finished document and does nothing.
    await ingest_document(document_id, session_factory=factory)
    await assert_document_state(factory, document_id, DocumentStatus.COMPLETED, 1)
    assert load.await_count == 1


async def test_duplicate_takes_over_when_active_attempt_fails(
    committed_document, fake_pipeline
):
    factory, document_id = committed_document
    _, graph = fake_pipeline
    graph.side_effect = ConnectionError("graph service unavailable")
    with pytest.raises(ConnectionError):
        await ingest_document(document_id, session_factory=factory)
    # The lock is gone with the failed attempt, so the re-queued copy can finish.
    graph.side_effect = None
    await ingest_document(document_id, session_factory=factory)
    await assert_document_state(factory, document_id, DocumentStatus.COMPLETED, 1)


async def count_checkpoints(factory, document_id):
    async with factory() as check:
        return await check.scalar(
            select(func.count())
            .select_from(IngestionCheckpoint)
            .where(IngestionCheckpoint.document_id == document_id)
        )


@pytest.fixture
def graph_pipeline_with_fake_models():
    """Real graph and community code against PostgreSQL; only model calls are fake."""
    parents = [{"chunk_index": 0, "content": "Tim Cook is the CEO of Apple."}]
    children = [{**parents[0], "parent_index": 0, "embedding": [0.1] * 768}]
    extraction = ExtractionResult(
        entities=[
            ExtractedEntity(name="Apple", type="COMPANY"),
            ExtractedEntity(name="Tim Cook", type="PERSON"),
        ],
        relationships=[
            ExtractedRelationship(
                source="Tim Cook", target="Apple", relationship_type="CEO_OF"
            )
        ],
    )
    summary_llm = MagicMock()
    summary_llm.ainvoke = AsyncMock(
        return_value=CommunitySummary(title="Apple leadership", summary="Tim Cook.")
    )
    summary_embeddings = MagicMock()
    summary_embeddings.aembed_documents = AsyncMock(return_value=[[0.2] * 768])
    with (
        patch(f"{_MODULE}.load_pdf_documents", new=AsyncMock(return_value=[object()])),
        patch(
            f"{_MODULE}.split_into_parent_and_child_chunks",
            return_value=(parents, children),
        ),
        patch(
            f"{_MODULE}.generate_child_embeddings", new=AsyncMock(return_value=children)
        ),
        patch(
            "financial_assistant.ai.graph_extraction.extract_entities_and_relationships",
            new=AsyncMock(return_value=extraction),
        ) as extract,
        patch(
            "financial_assistant.ai.community_detection._structured_summarizer",
            summary_llm,
        ),
        patch(
            "financial_assistant.ai.community_detection._embeddings_model",
            summary_embeddings,
        ),
    ):
        yield extract, summary_llm.ainvoke, summary_embeddings.aembed_documents


async def test_retry_reuses_model_results_committed_by_failed_attempt(
    committed_document, graph_pipeline_with_fake_models
):
    factory, document_id = committed_document
    extract, summarize, embed_summaries = graph_pipeline_with_fake_models
    # Fail at the very last model call, after extraction and summary succeeded.
    embed_summaries.side_effect = ConnectionError("embedding unavailable")

    with pytest.raises(ConnectionError):
        await ingest_document(document_id, session_factory=factory)

    # The attempt rolled back, but its model results were committed separately while
    # it held the document lock (this would hang if the lock blocked the FK check).
    await assert_document_state(factory, document_id, DocumentStatus.PROCESSING, 0)
    assert await count_checkpoints(factory, document_id) == 2

    embed_summaries.side_effect = None
    embed_summaries.return_value = [[0.2] * 768]
    await ingest_document(document_id, session_factory=factory)

    await assert_document_state(factory, document_id, DocumentStatus.COMPLETED, 1)
    assert extract.await_count == 1
    assert summarize.await_count == 1
    assert embed_summaries.await_count == 2
    # Completing the document removes its checkpoints in the same transaction.
    assert await count_checkpoints(factory, document_id) == 0


async def test_checkpoints_are_deleted_with_their_document(committed_document):
    factory, document_id = committed_document
    await AttemptCheckpoints(document_id, factory).put(
        CheckpointKind.GRAPH_EXTRACTION, "key", {"ok": True}
    )
    assert await AttemptCheckpoints(document_id, factory).get(
        CheckpointKind.GRAPH_EXTRACTION, "key"
    ) == {"ok": True}
    async with factory() as cleanup:
        await cleanup.execute(delete(Document).where(Document.id == document_id))
        await cleanup.commit()
    assert await count_checkpoints(factory, document_id) == 0
