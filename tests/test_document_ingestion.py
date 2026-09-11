import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from financial_assistant.ai.document_ingestion import (
    generate_child_embeddings,
    process_uploaded_document,
)
from financial_assistant.models.document import Document, DocumentStatus


def _session_context(session: MagicMock) -> MagicMock:
    """Wrap a mock session so it can be returned by an async context manager."""
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=session)
    # Returning False ensures exceptions raised inside `async with` are propagated.
    context.__aexit__ = AsyncMock(return_value=False)
    return context


@pytest.fixture
def ingestion_data():
    """Provide the smallest valid parent/child chunk hierarchy for ingestion."""
    parent_chunks = [{"chunk_index": 0, "content": "parent"}]
    child_chunks = [
        {
            "parent_index": 0,
            "chunk_index": 0,
            "content": "child",
            "embedding": [0.1],
        }
    ]
    return parent_chunks, child_chunks


async def test_processing_failure_rolls_back_before_recording_failed_status(
    ingestion_data,
    caplog,
):
    """A late failure should roll back partial data and persist FAILED separately."""
    parent_chunks, child_chunks = ingestion_data
    processing_session = MagicMock()
    processing_session.get = AsyncMock(
        return_value=MagicMock(status=DocumentStatus.PROCESSING)
    )
    processing_session.flush = AsyncMock()
    processing_session.rollback = AsyncMock()
    processing_session.commit = AsyncMock()

    failed_document = MagicMock(status=DocumentStatus.PROCESSING)
    status_session = MagicMock()
    status_session.get = AsyncMock(return_value=failed_document)
    status_session.commit = AsyncMock()

    # The first factory call supplies the processing session; after rollback and
    # closure, the second supplies an independent session for the FAILED update.
    session_maker = MagicMock(
        side_effect=[
            _session_context(processing_session),
            _session_context(status_session),
        ]
    )

    # Keep setup deterministic and inject the failure after both chunk collections
    # have been flushed.
    with (
        patch(
            "financial_assistant.ai.document_ingestion.async_session_maker",
            session_maker,
        ),
        patch(
            "financial_assistant.ai.document_ingestion.load_pdf_documents",
            new=AsyncMock(return_value=[MagicMock()]),
        ),
        patch(
            "financial_assistant.ai.document_ingestion.split_into_parent_and_child_chunks",
            return_value=(parent_chunks, child_chunks),
        ),
        patch(
            "financial_assistant.ai.document_ingestion.generate_child_embeddings",
            new=AsyncMock(return_value=child_chunks),
        ),
        patch(
            "financial_assistant.ai.document_ingestion.process_document_graph",
            new=AsyncMock(side_effect=RuntimeError("graph processing failed")),
        ),
    ):
        await process_uploaded_document(document_id=42)

    # Partial chunk data must be rolled back, while only the FAILED status is
    # committed through the clean session.
    assert session_maker.call_count == 2
    processing_session.rollback.assert_awaited_once_with()
    processing_session.commit.assert_not_awaited()
    processing_session.get.assert_awaited_once_with(Document, 42, with_for_update=True)
    status_session.get.assert_awaited_once_with(Document, 42, with_for_update=True)
    assert failed_document.status == DocumentStatus.FAILED
    status_session.commit.assert_awaited_once_with()
    error_record = next(
        record
        for record in caplog.records
        if record.getMessage() == "Error processing document 42"
    )
    assert error_record.exc_info is not None


async def test_flush_failure_uses_fresh_session_to_record_failed_status(
    ingestion_data,
):
    """A flush failure should not reuse the failed session to persist FAILED."""
    parent_chunks, child_chunks = ingestion_data
    processing_session = MagicMock()
    processing_session.get = AsyncMock(
        return_value=MagicMock(status=DocumentStatus.PROCESSING)
    )
    # A failed flush leaves a real SQLAlchemy session unusable until it is rolled back.
    processing_session.flush = AsyncMock(side_effect=RuntimeError("flush failed"))
    processing_session.rollback = AsyncMock()
    processing_session.commit = AsyncMock()

    failed_document = MagicMock(status=DocumentStatus.PROCESSING)
    status_session = MagicMock()
    status_session.get = AsyncMock(return_value=failed_document)
    status_session.commit = AsyncMock()

    # The first factory call supplies the failed processing session; the second
    # supplies a clean session that can safely persist the status update.
    session_maker = MagicMock(
        side_effect=[
            _session_context(processing_session),
            _session_context(status_session),
        ]
    )

    # No later processing functions need patches because execution stops at the first
    # parent-chunk flush.
    with (
        patch(
            "financial_assistant.ai.document_ingestion.async_session_maker",
            session_maker,
        ),
        patch(
            "financial_assistant.ai.document_ingestion.load_pdf_documents",
            new=AsyncMock(return_value=[MagicMock()]),
        ),
        patch(
            "financial_assistant.ai.document_ingestion.split_into_parent_and_child_chunks",
            return_value=(parent_chunks, child_chunks),
        ),
        patch(
            "financial_assistant.ai.document_ingestion.generate_child_embeddings",
            new=AsyncMock(return_value=child_chunks),
        ),
    ):
        await process_uploaded_document(document_id=42)

    # Status lookup and commit must happen only through the fresh session.
    assert session_maker.call_count == 2
    processing_session.rollback.assert_awaited_once_with()
    processing_session.get.assert_awaited_once_with(Document, 42, with_for_update=True)
    processing_session.commit.assert_not_awaited()
    status_session.get.assert_awaited_once_with(Document, 42, with_for_update=True)
    assert failed_document.status == DocumentStatus.FAILED
    status_session.commit.assert_awaited_once_with()


async def test_successful_processing_commits_completed_status(ingestion_data):
    """Successful ingestion should commit COMPLETED without opening a second session."""
    parent_chunks, child_chunks = ingestion_data
    completed_document = MagicMock(status=DocumentStatus.PROCESSING)
    processing_session = MagicMock()
    processing_session.get = AsyncMock(
        return_value=MagicMock(status=DocumentStatus.PROCESSING)
    )
    processing_session.flush = AsyncMock()
    processing_session.rollback = AsyncMock()
    processing_session.get = AsyncMock(return_value=completed_document)
    processing_session.commit = AsyncMock()
    session_maker = MagicMock(return_value=_session_context(processing_session))

    with (
        patch(
            "financial_assistant.ai.document_ingestion.async_session_maker",
            session_maker,
        ),
        patch(
            "financial_assistant.ai.document_ingestion.document_file_path",
            return_value=Path("/shared/documents/42.pdf"),
        ),
        patch(
            "financial_assistant.ai.document_ingestion.load_pdf_documents",
            return_value=[MagicMock()],
        ) as load_pdf,
        patch(
            "financial_assistant.ai.document_ingestion.split_into_parent_and_child_chunks",
            return_value=(parent_chunks, child_chunks),
        ),
        patch(
            "financial_assistant.ai.document_ingestion.generate_child_embeddings",
            return_value=child_chunks,
        ),
        patch(
            "financial_assistant.ai.document_ingestion.process_document_graph",
            return_value=[],
        ) as process_graph,
        patch(
            "financial_assistant.ai.document_ingestion.process_document_communities"
        ) as process_communities,
    ):
        await process_uploaded_document(document_id=42)
    load_pdf.assert_awaited_once_with(Path("/shared/documents/42.pdf"))

    session_maker.assert_called_once_with()
    assert processing_session.flush.await_count == 2
    processing_session.get.assert_awaited_once_with(Document, 42, with_for_update=True)
    assert completed_document.status == DocumentStatus.COMPLETED
    processing_session.commit.assert_awaited_once_with()
    processing_session.rollback.assert_not_awaited()
    process_graph.assert_awaited_once()
    process_communities.assert_awaited_once_with(processing_session, 42, [])


async def test_generate_child_embeddings_rejects_empty_input():
    """An empty embedding request should fail before contacting the provider."""
    embeddings_model = MagicMock()
    embeddings_model.aembed_documents = AsyncMock()

    with (
        patch(
            "financial_assistant.ai.document_ingestion.embeddings_model",
            embeddings_model,
        ),
        pytest.raises(ValueError, match="without child chunks"),
    ):
        await generate_child_embeddings([])

    embeddings_model.aembed_documents.assert_not_awaited()


async def test_generate_child_embeddings_rejects_count_mismatch():
    """Every child chunk must receive exactly one embedding."""
    child_chunks = [{"content": "child"}]

    embeddings_model = MagicMock()
    embeddings_model.aembed_documents = AsyncMock(return_value=[])

    with (
        patch(
            "financial_assistant.ai.document_ingestion.embeddings_model",
            embeddings_model,
        ),
        pytest.raises(ValueError, match="Embedding count does not match"),
    ):
        await generate_child_embeddings(child_chunks)

    embeddings_model.aembed_documents.assert_awaited_once_with(["child"])
    assert "embedding" not in child_chunks[0]


@pytest.mark.parametrize(
    ("parent_chunks", "child_chunks", "error_message"),
    [
        ([], [], "PDF produced no parent chunks"),
        (
            [{"chunk_index": 0, "content": "parent"}],
            [],
            "PDF produced no child chunks",
        ),
    ],
)
async def test_empty_chunk_results_mark_document_failed(
    parent_chunks,
    child_chunks,
    error_message,
    caplog,
):
    """A PDF that produces no usable chunks must not be marked COMPLETED."""
    processing_session = MagicMock()
    processing_session.get = AsyncMock(
        return_value=MagicMock(status=DocumentStatus.PROCESSING)
    )
    processing_session.rollback = AsyncMock()
    processing_session.commit = AsyncMock()

    failed_document = MagicMock(status=DocumentStatus.PROCESSING)
    status_session = MagicMock()
    status_session.get = AsyncMock(return_value=failed_document)
    status_session.commit = AsyncMock()
    session_maker = MagicMock(
        side_effect=[
            _session_context(processing_session),
            _session_context(status_session),
        ]
    )

    with (
        patch(
            "financial_assistant.ai.document_ingestion.async_session_maker",
            session_maker,
        ),
        patch(
            "financial_assistant.ai.document_ingestion.load_pdf_documents",
            return_value=[MagicMock()],
        ),
        patch(
            "financial_assistant.ai.document_ingestion.split_into_parent_and_child_chunks",
            return_value=(parent_chunks, child_chunks),
        ),
        patch(
            "financial_assistant.ai.document_ingestion.generate_child_embeddings"
        ) as generate_embeddings,
    ):
        await process_uploaded_document(document_id=42)

    processing_session.rollback.assert_awaited_once_with()
    processing_session.commit.assert_not_awaited()
    generate_embeddings.assert_not_awaited()
    assert failed_document.status == DocumentStatus.FAILED
    status_session.commit.assert_awaited_once_with()
    error_record = next(
        record
        for record in caplog.records
        if record.getMessage() == "Error processing document 42"
    )
    assert str(error_record.exc_info[1]) == error_message


async def test_pdf_without_readable_pages_marks_document_failed(caplog):
    """An empty PDF-loader result must stop ingestion before chunking."""
    processing_session = MagicMock()
    processing_session.get = AsyncMock(
        return_value=MagicMock(status=DocumentStatus.PROCESSING)
    )
    processing_session.rollback = AsyncMock()

    failed_document = MagicMock(status=DocumentStatus.PROCESSING)
    status_session = MagicMock()
    status_session.get = AsyncMock(return_value=failed_document)
    status_session.commit = AsyncMock()
    session_maker = MagicMock(
        side_effect=[
            _session_context(processing_session),
            _session_context(status_session),
        ]
    )

    with (
        patch(
            "financial_assistant.ai.document_ingestion.async_session_maker",
            session_maker,
        ),
        patch(
            "financial_assistant.ai.document_ingestion.load_pdf_documents",
            return_value=[],
        ),
        patch(
            "financial_assistant.ai.document_ingestion.split_into_parent_and_child_chunks"
        ) as split_chunks,
    ):
        await process_uploaded_document(document_id=42)

    processing_session.rollback.assert_awaited_once_with()
    split_chunks.assert_not_called()
    assert failed_document.status == DocumentStatus.FAILED
    error_record = next(
        record
        for record in caplog.records
        if record.getMessage() == "Error processing document 42"
    )
    assert str(error_record.exc_info[1]) == "PDF contains no readable pages"


async def test_cancellation_rolls_back_marks_failed_and_propagates(caplog):
    """Cancellation should clean up state while remaining visible to the caller."""
    processing_session = MagicMock()
    processing_session.get = AsyncMock(
        return_value=MagicMock(status=DocumentStatus.PROCESSING)
    )
    processing_session.rollback = AsyncMock()

    failed_document = MagicMock(status=DocumentStatus.PROCESSING)
    status_session = MagicMock()
    status_session.get = AsyncMock(return_value=failed_document)
    status_session.commit = AsyncMock()
    session_maker = MagicMock(
        side_effect=[
            _session_context(processing_session),
            _session_context(status_session),
        ]
    )

    with (
        patch(
            "financial_assistant.ai.document_ingestion.async_session_maker",
            session_maker,
        ),
        patch(
            "financial_assistant.ai.document_ingestion.load_pdf_documents",
            side_effect=asyncio.CancelledError,
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        await process_uploaded_document(document_id=42)

    processing_session.rollback.assert_awaited_once_with()
    assert failed_document.status == DocumentStatus.FAILED
    status_session.commit.assert_awaited_once_with()
    cancellation_record = next(
        record
        for record in caplog.records
        if record.getMessage() == "Processing cancelled for document 42"
    )
    assert cancellation_record.exc_info is not None
