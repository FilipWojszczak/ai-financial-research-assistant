import asyncio
import tempfile
import uuid
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document as LangchainDocument


async def load_pdf_documents(file_bytes: bytes) -> list[LangchainDocument]:
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
