"""Settle failed ingestion messages after committing their document status."""

import argparse
import asyncio
import logging
import signal
import threading
import uuid
from contextlib import closing

from sqlalchemy import select
from sqlalchemy.exc import DBAPIError

from .core.celery_app import celery_app
from .core.db import async_session_maker, engine
from .core.messaging import INGESTION_TASK_NAME, dead_letter_queue
from .models import Document, DocumentOutbox
from .models.document import DocumentStatus

logger = logging.getLogger(__name__)


async def mark_failed_task(task_id: uuid.UUID) -> bool:
    """Return False for active ingestion; retain delivery on database errors."""
    changed = False
    try:
        async with (
            asyncio.timeout(5),
            async_session_maker() as session,
            session.begin(),
        ):
            # The outbox is authoritative: never trust a document ID from the payload.
            document_id = await session.scalar(
                select(DocumentOutbox.document_id).where(DocumentOutbox.id == task_id)
            )
            if document_id is None:
                # A deleted document's outbox disappears too; old messages are harmless.
                return True
            document = await session.get(
                Document, document_id, with_for_update={"nowait": True}
            )
            if document is not None and document.status == DocumentStatus.PROCESSING:
                document.status = DocumentStatus.FAILED
                changed = True
        # Leaving session.begin() committed before the caller acknowledges the message.
        if changed:
            logger.info(
                "Document %d: terminal failure for task %s", document_id, task_id
            )
        return True
    except DBAPIError as exc:
        # PostgreSQL 55P03 (lock_not_available): NOWAIT could not acquire the
        # lock because another transaction holds a conflicting lock.
        if getattr(exc.orig, "sqlstate", None) == "55P03":
            # Leave the message unacknowledged so reconciliation can retry later.
            return False
        # Propagate other database errors rather than treating them as lock contention.
        raise


def reconcile_once(
    runner: asyncio.Runner, *, limit: int = 100, stop: threading.Event | None = None
) -> int:
    if type(limit) is not int or limit <= 0:
        raise ValueError("limit must be a positive integer")
    settled = 0
    with celery_app.connection_for_read(
        connect_timeout=5,
        transport_options={"read_timeout": 5, "write_timeout": 5},
    ) as connection:
        connection.ensure_connection(max_retries=0)
        with closing(connection.channel()) as channel:
            queue = dead_letter_queue(channel)
            queue.declare()
            for _ in range(limit):
                if stop is not None and stop.is_set():
                    break
                message = queue.get(no_ack=False, accept=["json"])
                if message is None:
                    break
                try:
                    if message.headers.get("task") != INGESTION_TASK_NAME:
                        raise ValueError("Unexpected task")
                    task_id = uuid.UUID(message.headers["id"])
                except (ValueError, TypeError, KeyError, AttributeError):
                    # Keep unfamiliar deliveries for manual inspection, without
                    # logging their potentially sensitive payloads.
                    logger.error(
                        "Unrecognized message retained in ingestion failure queue"
                    )
                    continue
                if runner.run(mark_failed_task(task_id)):
                    message.ack()
                    settled += 1
                # Busy and unrecognized messages stay unacknowledged until channel
                # close. This allows later messages through without a hot requeue loop.
    return settled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    logging.basicConfig(level=logging.INFO)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    with asyncio.Runner() as runner:
        try:
            while not stop.is_set():
                try:
                    count = reconcile_once(runner, limit=args.limit, stop=stop)
                    if count:
                        logger.info("Settled %d ingestion failures", count)
                except Exception as exc:
                    if not args.loop:
                        raise
                    logger.error(
                        "Failure reconciliation postponed (%s)", type(exc).__name__
                    )
                if not args.loop:
                    break
                stop.wait(2)
        finally:
            runner.run(engine.dispose())


if __name__ == "__main__":
    main()
