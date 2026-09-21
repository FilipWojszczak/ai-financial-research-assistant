"""Celery entry point for source PDFs already stored by the API."""

import logging

from celery.exceptions import Reject
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from ..ai.document_ingestion import (
    InvalidDocumentError,
    _mark_document_failed,
    ingest_document,
)
from ..core.celery_app import celery_app
from ..core.config import get_settings
from ..core.messaging import INGESTION_TASK_NAME
from .async_runner import get_worker_runner

logger = logging.getLogger(__name__)


async def _run_attempt(document_id: int, *, final_attempt: bool) -> None:
    # Own the database resources for this attempt. Never reuse the API's pool
    # or any connections inherited from a prefork parent.
    engine = create_async_engine(get_settings().database_url, poolclass=NullPool)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        try:
            await ingest_document(document_id, session_factory=session_factory)
        except Exception as exc:
            if final_attempt or isinstance(exc, InvalidDocumentError):
                await _mark_document_failed(
                    document_id, session_factory=session_factory
                )
            raise
    finally:
        await engine.dispose()


@celery_app.task(
    bind=True,
    name=INGESTION_TASK_NAME,
    acks_late=True,
    reject_on_worker_lost=True,
    acks_on_failure_or_timeout=True,
    max_retries=3,
)
def ingest_document_task(self, document_id: int) -> None:
    """Process an ID with up to three retries (four attempts in total)."""
    if type(document_id) is not int or document_id <= 0:
        raise Reject("document_id must be a positive integer", requeue=False)

    final_attempt = self.request.retries >= self.max_retries
    try:
        get_worker_runner().run(_run_attempt(document_id, final_attempt=final_attempt))
    except InvalidDocumentError as exc:
        logger.exception(
            "Invalid document %d; rejecting to dead-letter queue", document_id
        )
        raise Reject(str(exc), requeue=False) from exc
    except Exception as exc:
        if final_attempt:
            logger.exception(
                "Document %d exhausted retries; rejecting to DLQ", document_id
            )
            raise Reject(str(exc), requeue=False) from exc
        # Retryable failures remain PROCESSING while Celery schedules a new message.
        # request.retries starts at zero: delays are 10, 20 and 40 seconds.
        raise self.retry(exc=exc, countdown=10 * (2**self.request.retries)) from exc
