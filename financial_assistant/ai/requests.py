"""Bound individual AI operations without limiting a whole document's runtime."""

import asyncio
import logging
from collections.abc import Awaitable

from langchain_core.embeddings import Embeddings

from ..core.config import get_settings

logger = logging.getLogger(__name__)
_EMBEDDING_BATCH_SIZE = 16


async def ai_request[T](request: Awaitable[T]) -> T:
    # Cancellation finishes before control returns to the persistent worker loop.
    async with asyncio.timeout(get_settings().ai_request_timeout_seconds):
        return await request


async def embed_texts(model: Embeddings, texts: list[str]) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), _EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + _EMBEDDING_BATCH_SIZE]
        result = await ai_request(model.aembed_documents(batch))
        if len(result) != len(batch):
            raise ValueError("Embedding count does not match input count")
        embeddings.extend(result)
        logger.info("Embedded %d/%d texts", len(embeddings), len(texts))
    return embeddings
