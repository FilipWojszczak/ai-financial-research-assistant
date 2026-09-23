"""Periodically delete old, unreferenced source PDFs and abandoned upload files."""

import argparse
import asyncio
import logging
import math
import re
import signal
import stat
import time
from contextlib import suppress

from sqlalchemy import select

from .core.config import get_settings
from .core.db import async_session_maker, engine
from .core.document_storage import lock_document_storage
from .models import Document

logger = logging.getLogger(__name__)
_SOURCE = re.compile(r"([1-9][0-9]*)\.pdf")
_TEMPORARY = re.compile(r"\.([1-9][0-9]*)\.pdf\.[0-9a-f]{32}\.tmp")


async def cleanup_orphaned_files(*, min_age_seconds: float = 86400) -> int:
    if not math.isfinite(min_age_seconds) or min_age_seconds <= 0:
        raise ValueError("min_age_seconds must be positive and finite")
    directory = get_settings().document_storage_path
    if not directory.exists():
        return 0
    cutoff = time.time() - min_age_seconds
    removed = 0
    for path in directory.iterdir():
        temporary = _TEMPORARY.fullmatch(path.name)
        match = temporary or _SOURCE.fullmatch(path.name)
        if match is None:
            continue
        document_id = int(match[1])
        if document_id > 2**31 - 1:
            continue
        try:
            before = path.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(before.st_mode) or before.st_mtime >= cutoff:
            continue
        async with async_session_maker() as session, session.begin():
            # Upload holds the same transaction-level lock until COMMIT/ROLLBACK.
            # A file for an uncommitted upload must never look like an orphan.
            if not await lock_document_storage(session, document_id, wait=False):
                continue
            exists = await session.scalar(
                select(Document.id).where(Document.id == document_id)
            )
            if exists is not None and temporary is None:
                continue
            try:
                after = path.lstat()
            except FileNotFoundError:
                continue
            if (
                not stat.S_ISREG(after.st_mode)
                or after.st_mtime >= cutoff
                or (before.st_dev, before.st_ino, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_mtime_ns)
            ):
                continue
            # A DB error above aborts this pass before deletion. Keep the upload
            # lock while unlinking, including if this coroutine is cancelled.
            deletion = asyncio.create_task(
                asyncio.to_thread(path.unlink, missing_ok=True)
            )
            try:
                await asyncio.shield(deletion)
            except asyncio.CancelledError:
                await deletion
                raise
            removed += 1
            logger.info("Removed orphaned upload file %s", path.name)
    return removed


async def _main(*, continuous: bool, min_age: float, interval: float) -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    try:
        while not stop.is_set():
            try:
                await cleanup_orphaned_files(min_age_seconds=min_age)
            except Exception as exc:
                if not continuous:
                    raise
                logger.error("Storage cleanup postponed (%s)", type(exc).__name__)
            if not continuous or stop.is_set():
                break
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=interval)
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--min-age-seconds", type=float, default=86400)
    parser.add_argument("--interval", type=float, default=3600)
    args = parser.parse_args()
    for value in (args.min_age_seconds, args.interval):
        if not math.isfinite(value) or value <= 0:
            parser.error("Age and interval must be positive and finite")
    logging.basicConfig(level=logging.INFO)
    asyncio.run(
        _main(
            continuous=args.loop, min_age=args.min_age_seconds, interval=args.interval
        )
    )
