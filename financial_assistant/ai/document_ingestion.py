import asyncio
import logging
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document as LangchainDocument
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..core.db import async_session_maker
from ..core.document_storage import document_file_path
from ..models.document import ChildChunk, Document, DocumentStatus, ParentChunk
from .community_detection import process_document_communities
from .graph_extraction import process_document_graph

logger = logging.getLogger(__name__)

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", output_dimensionality=768
)


async def load_pdf_documents(file_path: Path) -> list[LangchainDocument]:
    """Load a stored PDF without blocking the application's event loop."""
    loader = PyPDFLoader(str(file_path))
    return await asyncio.to_thread(loader.load)


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
    embeddings = await embeddings_model.aembed_documents(texts)

    if len(embeddings) != len(child_chunks):
        raise ValueError(
            "Embedding count does not match child chunk count: "
            f"{len(embeddings)} != {len(child_chunks)}"
        )

    # Attach the generated embeddings back to the child chunks
    for chunk, embedding in zip(child_chunks, embeddings, strict=True):
        chunk["embedding"] = embedding

    return child_chunks


async def _mark_document_failed(document_id: int) -> None:
    """Persist FAILED independently from the rolled-back processing transaction."""
    try:
        async with async_session_maker() as status_session:
            document = await status_session.get(Document, document_id)
            if document:
                document.status = DocumentStatus.FAILED
                await status_session.commit()
    except Exception:
        logger.exception(
            "Failed to update status to FAILED for document %d", document_id
        )


async def process_uploaded_document(document_id: int) -> None:
    try:
        async with async_session_maker() as session:
            try:
                documents = await load_pdf_documents(document_file_path(document_id))
                if not documents:
                    raise ValueError("PDF contains no readable pages")

                parent_chunks, child_chunks = split_into_parent_and_child_chunks(
                    documents
                )
                if not parent_chunks:
                    raise ValueError("PDF produced no parent chunks")
                if not child_chunks:
                    raise ValueError("PDF produced no child chunks")

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

                # Extract entities/relationships and build the knowledge graph
                entities = await process_document_graph(
                    session, document_id, db_parents
                )
                await process_document_communities(session, document_id, entities)

                # Update document status to COMPLETED
                document = await session.get(Document, document_id)
                if document:
                    document.status = DocumentStatus.COMPLETED
                await session.commit()
            except BaseException:
                # Roll back failures and cancellation before closing the session.
                await session.rollback()
                raise
    except asyncio.CancelledError:
        logger.warning(
            "Processing cancelled for document %d", document_id, exc_info=True
        )
        await _mark_document_failed(document_id)
        raise
    except Exception:
        # Record FAILED independently because the processing session may be unusable.
        logger.exception("Error processing document %d", document_id)
        await _mark_document_failed(document_id)
