import asyncio
import os
import uuid
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .config import get_settings

# This can be any integer; 0x504446 was chosen because it encodes "PDF" in ASCII.
_STORAGE_LOCK_NAMESPACE = 0x504446


async def lock_document_storage(
    session: AsyncSession, document_id: int, *, wait: bool = True
) -> bool:
    """Coordinate file writes and cleanup until the owning DB transaction ends."""
    if type(document_id) is not int or document_id <= 0:
        raise ValueError("document_id must be a positive integer")
    lock = func.pg_advisory_xact_lock if wait else func.pg_try_advisory_xact_lock
    result = await session.scalar(select(lock(_STORAGE_LOCK_NAMESPACE, document_id)))
    return True if wait else bool(result)


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
        # Persist the renamed directory entry as well as the file contents.
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


async def store_document_file(document_id: int, file_bytes: bytes) -> Path:
    """Persist a source PDF and return its stable path."""
    destination = document_file_path(document_id)
    write = asyncio.create_task(
        asyncio.to_thread(_write_atomically, destination, file_bytes)
    )
    try:
        await asyncio.shield(write)
    except asyncio.CancelledError:
        # Do not release an upload's DB lock while its filesystem thread still writes.
        await write
        raise
    return destination


async def delete_document_file(document_id: int) -> None:
    """Delete a source PDF if it exists."""
    await asyncio.to_thread(document_file_path(document_id).unlink, missing_ok=True)
