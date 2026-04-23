import asyncio
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document as LangchainDocument
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..core.db import async_session_maker
from ..models.document import ChildChunk, Document, DocumentStatus, ParentChunk

logger = logging.getLogger(__name__)


async def load_pdf_documents(file_bytes: bytes) -> list[LangchainDocument]:
    """
    Load PDF documents from bytes using PyPDFLoader. This function writes the bytes
    to a temporary file and then loads it.
    """
    # Create a safe temporary file path for the uploaded PDF
    temp_dir = Path(tempfile.gettempdir())
    temp_path = temp_dir / f"{uuid.uuid4()}.pdf"

    try:
        # Write the uploaded file bytes to the temporary file in a separate thread
        await asyncio.to_thread(temp_path.write_bytes, file_bytes)

        # Load the PDF document using PyPDFLoader in a separate thread
        loader = PyPDFLoader(str(temp_path))
        documents = await asyncio.to_thread(loader.load)
        return documents
    finally:
        # Clean up the temporary file in a separate thread
        await asyncio.to_thread(temp_path.unlink, missing_ok=True)


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
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", output_dimensionality=768
    )
    # Extract the content from child chunks to generate embeddings
    texts = [chunk["content"] for chunk in child_chunks]
    embeddings = await embeddings_model.aembed_documents(texts)

    # Attach the generated embeddings back to the child chunks
    for chunk, embedding in zip(child_chunks, embeddings, strict=True):
        chunk["embedding"] = embedding

    return child_chunks


async def process_uploaded_document(document_id: int, file_bytes: bytes) -> None:
    async with async_session_maker() as session:
        try:
            documents = await load_pdf_documents(file_bytes)
            parent_chunks, child_chunks = split_into_parent_and_child_chunks(documents)
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
                # Find the corresponding parent chunk object in the database using the
                # parent_index from the child chunk data
                parent_object = db_parents[child["parent_index"]]
                db_children.append(
                    ChildChunk(
                        chunk_index=child["chunk_index"],
                        content=child["content"],
                        embedding=child["embedding"],
                        # Link the child chunk to its parent chunk using the parent's ID
                        # from the database (not parent_index from the child chunk data)
                        parent_id=parent_object.id,
                    )
                )
            session.add_all(db_children)

            # Update document status to COMPLETED
            document = await session.get(Document, document_id)
            if document:
                document.status = DocumentStatus.COMPLETED
            await session.commit()
        except Exception as e:
            # Log the error and update document status to FAILED
            logger.error(f"Error processing document {document_id}: {e!s}")
            document = await session.get(Document, document_id)
            if document:
                document.status = DocumentStatus.FAILED
                await session.commit()
