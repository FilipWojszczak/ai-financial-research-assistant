import asyncio
import os
import uuid
from pathlib import Path

from .config import get_settings


def document_file_path(document_id: int) -> Path:
    """Return the server-controlled path for a document's source PDF."""
    if document_id <= 0:
        raise ValueError("document_id must be a positive integer")

    return get_settings().document_storage_path / f"{document_id}.pdf"


def _write_atomically(destination: Path, file_bytes: bytes) -> None:
    """Write bytes to a temporary sibling and atomically publish the final file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )

    try:
        with temporary_path.open("xb") as temporary_file:
            temporary_file.write(file_bytes)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


async def store_document_file(document_id: int, file_bytes: bytes) -> Path:
    """Persist a source PDF and return its stable path."""
    destination = document_file_path(document_id)
    await asyncio.to_thread(_write_atomically, destination, file_bytes)
    return destination


async def delete_document_file(document_id: int) -> None:
    """Delete a source PDF if it exists."""
    await asyncio.to_thread(document_file_path(document_id).unlink, missing_ok=True)
