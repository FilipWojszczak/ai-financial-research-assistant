"""Real Celery worker with deterministic provider responses for integration tests."""

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from celery import Task, current_task
from celery.signals import worker_ready
from langchain_core.messages import AIMessage

from financial_assistant.ai import (
    community_detection,
    document_ingestion,
    graph_extraction,
)
from financial_assistant.ai.graph_extraction import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)
from financial_assistant.core.celery_app import celery_app
from financial_assistant.tasks.document_ingestion import ingest_document_task

directory = Path(os.environ["INGESTION_TEST_DIRECTORY"])


async def embed(texts):
    return [[0.1] * 768 for _ in texts]


async def extract(prompt):
    task = cast(Task, current_task)
    with (directory / "attempts").open("a") as log:
        log.write(f"{task.request.retries}\n")
    if os.environ["INGESTION_TEST_FAILURE"] == "1":
        # Exercise the production per-call deadline, not a synthetic TimeoutError.
        await asyncio.Event().wait()
    return ExtractionResult(
        entities=[
            ExtractedEntity(name="Acme", type="COMPANY"),
            ExtractedEntity(name="Alice", type="PERSON"),
        ],
        relationships=[
            ExtractedRelationship(
                source="Alice", target="Acme", relationship_type="CEO_OF"
            )
        ],
    )


async def summarize(prompt):
    return AIMessage(content="TITLE: Acme leadership\nSUMMARY: Alice leads Acme.")


document_ingestion.embeddings_model = SimpleNamespace(aembed_documents=embed)
graph_extraction._structured_extractor = SimpleNamespace(ainvoke=extract)
community_detection._summary_llm = SimpleNamespace(ainvoke=summarize)
community_detection._embeddings_model = SimpleNamespace(aembed_documents=embed)

# Keep real broker retries while shortening the backoff for test runtime.
ingestion_task = cast(Task, ingest_document_task)
original_retry = ingestion_task.retry


def retry_quickly(*args, **kwargs):
    kwargs["countdown"] = 1
    return original_retry(*args, **kwargs)


ingestion_task.retry = retry_quickly


@worker_ready.connect
def record_ready(**kwargs):
    (directory / "ready").touch()


if __name__ == "__main__":
    celery_app.worker_main(
        [
            "worker",
            "--pool=prefork",
            "--concurrency=1",
            "--loglevel=INFO",
            "--without-gossip",
            "--without-mingle",
            "--without-heartbeat",
        ]
    )
