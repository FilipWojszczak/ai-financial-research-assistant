import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from financial_assistant.ai.document_ingestion import (
    InvalidDocumentError,
    generate_child_embeddings,
)
from financial_assistant.models.document import Document, DocumentStatus
from financial_assistant.tasks.document_ingestion import _run_attempt

PIPELINE = "financial_assistant.ai.document_ingestion"


@pytest.fixture
def attempt():
    processing = MagicMock()
    processing.get = AsyncMock(return_value=MagicMock(status=DocumentStatus.PROCESSING))
    processing.flush = AsyncMock()
    processing.rollback = AsyncMock()
    processing.commit = AsyncMock()
    status = MagicMock()
    status.get = AsyncMock(return_value=MagicMock(status=DocumentStatus.PROCESSING))
    status.commit = AsyncMock()

    def context(session):
        result = MagicMock()
        result.__aenter__ = AsyncMock(return_value=session)
        result.__aexit__ = AsyncMock(return_value=False)
        return result

    factory = MagicMock(side_effect=[context(processing), context(status)])
    engine = MagicMock(dispose=AsyncMock())
    with (
        patch(
            "financial_assistant.tasks.document_ingestion.create_async_engine",
            return_value=engine,
        ),
        patch(
            "financial_assistant.tasks.document_ingestion.async_sessionmaker",
            return_value=factory,
        ),
        patch(f"{PIPELINE}.document_file_path", return_value=Path("/shared/42.pdf")),
        patch(
            f"{PIPELINE}.load_pdf_documents", new=AsyncMock(return_value=[MagicMock()])
        ),
        patch(
            f"{PIPELINE}.split_into_parent_and_child_chunks",
            return_value=(
                [{"chunk_index": 0, "content": "parent"}],
                [{"parent_index": 0, "chunk_index": 0, "content": "child"}],
            ),
        ),
        patch(
            f"{PIPELINE}.generate_child_embeddings",
            new=AsyncMock(
                return_value=[
                    {
                        "parent_index": 0,
                        "chunk_index": 0,
                        "content": "child",
                        "embedding": [0.1],
                    }
                ]
            ),
        ),
        patch(f"{PIPELINE}.process_document_graph", new=AsyncMock(return_value=[])),
        patch(f"{PIPELINE}.process_document_communities", new=AsyncMock()),
    ):
        yield processing, status, factory, engine


@pytest.mark.parametrize("failure_at", ["graph", "flush"])
async def test_terminal_failure_rolls_back_before_fresh_status_update(
    attempt, failure_at
):
    processing, status, factory, engine = attempt
    events = []
    processing.rollback.side_effect = lambda: events.append("rollback")
    status.commit.side_effect = lambda: events.append("status_commit")
    error = RuntimeError("processing failed")
    if failure_at == "flush":
        processing.flush.side_effect = error
    with (
        patch(f"{PIPELINE}.process_document_graph", new=AsyncMock(side_effect=error)),
        pytest.raises(RuntimeError, match="processing failed"),
    ):
        await _run_attempt(42, final_attempt=True)

    assert events == ["rollback", "status_commit"]
    assert factory.call_count == 2
    processing.commit.assert_not_awaited()
    processing.get.assert_awaited_once_with(Document, 42, with_for_update=True)
    status.get.assert_awaited_once_with(Document, 42, with_for_update=True)
    assert status.get.return_value.status == DocumentStatus.FAILED
    engine.dispose.assert_awaited_once()


async def test_success_commits_completed_without_status_session(attempt):
    processing, status, factory, engine = attempt
    await _run_attempt(42, final_attempt=False)
    assert processing.get.return_value.status == DocumentStatus.COMPLETED
    assert processing.flush.await_count == 2
    processing.commit.assert_awaited_once()
    processing.rollback.assert_not_awaited()
    factory.assert_called_once()
    status.commit.assert_not_awaited()
    engine.dispose.assert_awaited_once()


@pytest.mark.parametrize(
    ("parents", "children", "message"),
    [
        ([], [], "PDF produced no parent chunks"),
        ([{"content": "parent"}], [], "PDF produced no child chunks"),
    ],
)
async def test_empty_chunks_are_permanent_failures(attempt, parents, children, message):
    processing, status, _, _ = attempt
    with (
        patch(
            f"{PIPELINE}.split_into_parent_and_child_chunks",
            return_value=(parents, children),
        ),
        pytest.raises(InvalidDocumentError, match=message),
    ):
        await _run_attempt(42, final_attempt=False)
    processing.rollback.assert_awaited_once()
    processing.commit.assert_not_awaited()
    assert status.get.return_value.status == DocumentStatus.FAILED


async def test_empty_pdf_is_a_permanent_failure(attempt):
    processing, status, _, _ = attempt
    with (
        patch(f"{PIPELINE}.load_pdf_documents", new=AsyncMock(return_value=[])),
        pytest.raises(InvalidDocumentError, match="PDF contains no readable pages"),
    ):
        await _run_attempt(42, final_attempt=False)
    processing.rollback.assert_awaited_once()
    assert status.get.return_value.status == DocumentStatus.FAILED


async def test_cancellation_rolls_back_and_allows_redelivery(attempt):
    processing, status, factory, _ = attempt
    with (
        patch(
            f"{PIPELINE}.load_pdf_documents",
            new=AsyncMock(side_effect=asyncio.CancelledError),
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        await _run_attempt(42, final_attempt=False)
    processing.rollback.assert_awaited_once()
    factory.assert_called_once()
    status.commit.assert_not_awaited()
    assert processing.get.return_value.status == DocumentStatus.PROCESSING


async def test_generate_child_embeddings_rejects_empty_input():
    model = MagicMock(aembed_documents=AsyncMock())
    with (
        patch(f"{PIPELINE}.embeddings_model", model),
        pytest.raises(ValueError, match="without child chunks"),
    ):
        await generate_child_embeddings([])
    model.aembed_documents.assert_not_awaited()


async def test_generate_child_embeddings_rejects_count_mismatch():
    chunks = [{"content": "child"}]
    model = MagicMock(aembed_documents=AsyncMock(return_value=[]))
    with (
        patch(f"{PIPELINE}.embeddings_model", model),
        pytest.raises(ValueError, match="Embedding count does not match"),
    ):
        await generate_child_embeddings(chunks)
    model.aembed_documents.assert_awaited_once_with(["child"])
    assert "embedding" not in chunks[0]
