"""Publish committed outbox requests; no broker call runs inside an API request."""

import asyncio
import logging
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ..models import DocumentOutbox
from .db import async_session_maker

logger = logging.getLogger(__name__)

INGESTION_TASK_NAME = "financial_assistant.tasks.document_ingestion.ingest_document"


def publish_ingestion(document_id: int, event_id: uuid.UUID) -> None:
    """Return only after RabbitMQ confirms publication, or raise an exception."""
    # Lazy import keeps broker configuration out of the API's storage transaction.
    from .celery_app import celery_app

    # A dedicated connection avoids waiting on a shared producer pool. Outbox
    # scheduling owns retries; do not hold a row lock through Celery retry sleeps.
    with celery_app.connection_for_write(
        connect_timeout=5,
        transport_options={
            "confirm_publish": True,
            "read_timeout": 5,
            "write_timeout": 5,
        },
    ) as connection:
        connection.ensure_connection(max_retries=0)
        with celery_app.amqp.Producer(connection) as producer:
            celery_app.send_task(
                INGESTION_TASK_NAME,
                kwargs={"document_id": document_id},
                task_id=str(event_id),
                queue="document_ingestion",
                producer=producer,
                retry=False,
                delivery_mode=2,
                timeout=5,
                confirm_timeout=5,
            )


async def publish_pending_documents(
    *,
    limit: int = 100,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    publish: Callable[[int, uuid.UUID], None] | None = None,
) -> int:
    """Attempt at most limit due rows, committing each independently.

    Returns the count confirmed and recorded as published. Failures stay pending
    with capped backoff. A crash after publish but before commit causes a resend.
    """
    if type(limit) is not int or limit <= 0:
        raise ValueError("limit must be a positive integer")
    factory = session_factory if session_factory is not None else async_session_maker
    send = publish if publish is not None else publish_ingestion
    published = 0
    for _ in range(limit):
        async with factory() as session, session.begin():
            event = await session.scalar(
                select(DocumentOutbox)
                .where(
                    DocumentOutbox.published_at.is_(None),
                    DocumentOutbox.next_attempt_at <= func.now(),
                )
                .order_by(DocumentOutbox.next_attempt_at, DocumentOutbox.created_at)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            if event is None:
                break
            event.attempts += 1
            try:
                # Synchronous AMQP I/O runs off the asyncio thread.
                await asyncio.to_thread(send, event.document_id, event.id)
            except Exception as exc:
                error_type = type(exc).__name__[:200]
                event.last_error = error_type
                delay = min(300, 5 * (2 ** min(event.attempts - 1, 6)))
                event.next_attempt_at = datetime.now(UTC) + timedelta(seconds=delay)
                logger.warning(
                    "Outbox %s publication failed (%s); retry in %ds",
                    event.id,
                    error_type,
                    delay,
                )
            else:
                event.published_at = datetime.now(UTC)
                event.last_error = None
                published += 1
    return published
