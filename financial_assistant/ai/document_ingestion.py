import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document as LangchainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
                    "parent_id": parent_index,  # Link child to its parent
                    "content": child_doc.page_content,
                    "chunk_index": child_index,
                }
            )

    return parent_chunks_data, child_chunks_data
