import asyncio
import logging
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document as LangchainDocument
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf.errors import PdfReadError
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ..core.document_storage import document_file_path
from ..models.document import ChildChunk, Document, DocumentStatus, ParentChunk
from .checkpoints import AttemptCheckpoints, delete_checkpoints
from .community_detection import process_document_communities
from .graph_extraction import process_document_graph
from .requests import embed_texts

logger = logging.getLogger(__name__)

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", output_dimensionality=768
)


# SQLAlchemy renders this as SELECT ... FOR NO KEY UPDATE NOWAIT on PostgreSQL.
INGESTION_LOCK = {"key_share": True, "nowait": True}
_LOCK_NOT_AVAILABLE = "55P03"


class InvalidDocumentError(ValueError):
    """The source PDF cannot be ingested by retrying the same input."""


class DocumentBusyError(RuntimeError):
    """Another attempt holds the document's lock; this delivery is a duplicate."""


async def load_pdf_documents(file_path: Path) -> list[LangchainDocument]:
    """Load a stored PDF without blocking the application's event loop."""

    def load():
        # Construct and run the loader in the worker thread, as construction may inspect
        # the filesystem.
        return PyPDFLoader(str(file_path)).load()

    try:
        return await asyncio.to_thread(load)
    except (FileNotFoundError, PdfReadError, ValueError) as exc:
        raise InvalidDocumentError(f"Cannot read source PDF: {file_path.name}") from exc


def split_into_parent_and_child_chunks(
    documents: list[LangchainDocument],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Split documents into parent and child chunks. Parent chunks are larger and provide
    broad context, while child chunks are smaller and are used for precise semantic
    meaning in vector search.
    """

    # Parent chunks: provide broad context for the LLM (e.g., a whole page or large
    #  section)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    # Child chunks: provide precise semantic meaning for vector search
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    parent_chunks_data = []
    child_chunks_data = []

    parent_docs = parent_splitter.split_documents(documents)
    for parent_index, parent_doc in enumerate(parent_docs):
        parent_chunks_data.append(
            {
                "content": parent_doc.page_content,
                "chunk_index": parent_index,
            }
        )

        # Split this specific parent into smaller children
        child_docs = child_splitter.split_documents([parent_doc])
        for child_index, child_doc in enumerate(child_docs):
            child_chunks_data.append(
                {
                    "parent_index": parent_index,  # Link child to its parent
                    "content": child_doc.page_content,
                    "chunk_index": child_index,
                }
            )

    return parent_chunks_data, child_chunks_data


async def generate_child_embeddings(
    child_chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Generate embeddings for child chunks using GoogleGenerativeAIEmbeddings. This
    function takes the child chunks, extracts their content, and generates embeddings.
    """
    if not child_chunks:
        raise ValueError("Cannot generate embeddings without child chunks")

    # Extract the content from child chunks to generate embeddings
    texts = [chunk["content"] for chunk in child_chunks]
    embeddings = await embed_texts(embeddings_model, texts)

    # Attach the generated embeddings back to the child chunks
    for chunk, embedding in zip(child_chunks, embeddings, strict=True):
        chunk["embedding"] = embedding

    return child_chunks


async def _mark_document_failed(
    document_id: int,
    *,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Best-effort terminal status update after the attempt has rolled back."""
    try:
        async with session_factory() as status_session:
            # Wait for a competing attempt and inspect its committed status. A late
            # failure must never overwrite another attempt's COMPLETED result.
            document = await status_session.get(
                Document, document_id, with_for_update=True
            )
            if document and document.status == DocumentStatus.PROCESSING:
                document.status = DocumentStatus.FAILED
                await status_session.commit()
    except Exception:
        # The failure consumer repeats this update after the message is parked.
        logger.exception(
            "Failed to update status to FAILED for document %d", document_id
        )


async def ingest_document(
    document_id: int,
    *,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Run one transactional attempt; propagate failures to the caller."""
    async with session_factory() as session:
        try:
            # Hold the row lock through the commit. A crash releases it on disconnect.
            # NO KEY UPDATE (not UPDATE) still excludes competing attempts, but lets
            # checkpoint rows referencing this document commit in their own sessions.
            # NOWAIT: a duplicate delivery must not occupy a worker process for the
            # whole length of the active attempt; the task re-queues it instead.
            try:
                document = await session.get(
                    Document, document_id, with_for_update=INGESTION_LOCK
                )
            except DBAPIError as exc:
                if getattr(exc.orig, "sqlstate", None) == _LOCK_NOT_AVAILABLE:
                    raise DocumentBusyError(
                        f"Document {document_id} is locked by another attempt"
                    ) from exc
                raise
            if document is None or document.status != DocumentStatus.PROCESSING:
                return

            logger.info("Document %d: loading source PDF", document_id)
            documents = await load_pdf_documents(document_file_path(document_id))
            if not documents:
                raise InvalidDocumentError("PDF contains no readable pages")

            parent_chunks, child_chunks = split_into_parent_and_child_chunks(documents)
            if not parent_chunks:
                raise InvalidDocumentError("PDF produced no parent chunks")
            if not child_chunks:
                raise InvalidDocumentError("PDF produced no child chunks")

            logger.info(
                "Document %d: embedding %d chunks", document_id, len(child_chunks)
            )
            embedded_children = await generate_child_embeddings(child_chunks)

            # Save parent_chunks and embedded_children to the database
            db_parents = [
                ParentChunk(
                    chunk_index=parent["chunk_index"],
                    content=parent["content"],
                    document_id=document_id,
                )
                for parent in parent_chunks
            ]
            session.add_all(db_parents)
            await session.flush()

            db_children = []
            for child in embedded_children:
                # Find the corresponding parent chunk object in the database using
                # the parent_index from the child chunk data
                parent_object = db_parents[child["parent_index"]]
                db_children.append(
                    ChildChunk(
                        chunk_index=child["chunk_index"],
                        content=child["content"],
                        embedding=child["embedding"],
                        # Link the child to the database parent ID, not its
                        # in-memory parent_index.
                        parent_id=parent_object.id,
                    )
                )
            session.add_all(db_children)
            await session.flush()

            # Model results from earlier failed attempts are reused; new ones are
            # committed as they arrive, so a retry does not repeat them.
            checkpoints = AttemptCheckpoints(document_id, session_factory)
            # Extract entities/relationships and build the knowledge graph
            entities = await process_document_graph(
                session, document_id, db_parents, checkpoints
            )
            await process_document_communities(
                session, document_id, entities, checkpoints
            )
            if checkpoints.hits:
                logger.info(
                    "Document %d: reused %d results from earlier attempts",
                    document_id,
                    checkpoints.hits,
                )

            # Checkpoints are only needed until the document completes.
            await delete_checkpoints(session, document_id)
            # Update document status to COMPLETED
            document.status = DocumentStatus.COMPLETED
            await session.commit()
            logger.info("Document %d: completed", document_id)
        except BaseException:
            # Includes cancellation: partial chunks and graph data must roll back.
            await session.rollback()
            raise
