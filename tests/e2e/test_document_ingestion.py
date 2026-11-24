"""Two complete ingestion flows using PostgreSQL, RabbitMQ, and a worker process."""

import os

import pytest
from sqlalchemy import func, select

from financial_assistant.models import Document, DocumentOutbox
from financial_assistant.models.document import ChildChunk, ParentChunk
from financial_assistant.models.graph import (
    Entity,
    EntityRelationship,
    GraphCommunity,
    GraphCommunityMembership,
)
from tests.e2e.flow import pdf_bytes, wait_until

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.getenv("RUN_INGESTION_TESTS") != "1",
        reason="Requires RUN_INGESTION_TESTS=1, PostgreSQL, and RabbitMQ management",
    ),
]


async def test_complete_pdf_ingestion(flow):
    document_id = await flow.upload()
    await flow.wait_status(document_id, "completed")
    async with flow.factory() as session:
        event = await session.scalar(
            select(DocumentOutbox).where(DocumentOutbox.document_id == document_id)
        )
        assert event.published_at is not None
        assert event.attempts == 1
        for model, expected in (
            (ParentChunk, 1),
            (ChildChunk, 1),
            (Entity, 2),
            (EntityRelationship, 1),
            (GraphCommunity, 1),
            (GraphCommunityMembership, 2),
        ):
            assert (
                await session.scalar(select(func.count()).select_from(model))
                == expected
            )
        child = await session.scalar(select(ChildChunk))
        community = await session.scalar(select(GraphCommunity))
        assert len(child.embedding) == len(community.embedding) == 768
    assert (flow.storage / f"{document_id}.pdf").read_bytes() == pdf_bytes()
    assert (flow.directory / "attempts").read_text().splitlines() == ["0"]
    assert await flow.queue_count() == 0


async def test_ai_timeout_exhausts_retries_and_finishes_failed(flow):
    document_id = await flow.upload(failure=True)
    await flow.wait_status(document_id, "failed")
    # Each task attempt sends the stalled call AI_REQUEST_ATTEMPTS (3) times before
    # the timeout fails the attempt; Celery then retries the task three times.
    assert (flow.directory / "attempts").read_text().splitlines() == [
        str(task_retry) for task_retry in range(4) for _ in range(3)
    ]
    assert (flow.storage / f"{document_id}.pdf").exists()
    async with flow.factory() as session:
        for model in (
            ParentChunk,
            ChildChunk,
            Entity,
            EntityRelationship,
            GraphCommunity,
            GraphCommunityMembership,
        ):
            assert await session.scalar(select(func.count()).select_from(model)) == 0
        event = await session.scalar(select(DocumentOutbox))
        assert event.published_at is not None

    async def parked():
        return await flow.queue_count() == 1

    await wait_until(parked, description="parked failure", processes=flow.processes)
    await flow.start("financial_assistant.services.reconcile_ingestion", "--loop")

    async def settled():
        return await flow.queue_count() == 0

    await wait_until(settled, description="settled failure", processes=flow.processes)
    async with flow.factory() as session:
        assert (await session.get(Document, document_id)).status == "failed"
