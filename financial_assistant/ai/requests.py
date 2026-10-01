"""Bound individual AI operations without limiting a whole document's runtime."""

import asyncio
import logging
from collections.abc import Awaitable

from google.genai.errors import ClientError
from langchain_core.embeddings import Embeddings

from ..core.config import get_settings

logger = logging.getLogger(__name__)
_EMBEDDING_BATCH_SIZE = 16
# Client errors that can succeed later: timeout, conflict, rate limit / quota.
_RETRYABLE_CLIENT_CODES = frozenset({408, 409, 429})


def is_permanent_provider_error(exc: BaseException) -> bool:
    """
    True when the model provider rejected the request itself (HTTP 4xx such as
    400 INVALID_ARGUMENT or 401/403). Sending the same request again cannot help,
    so retrying would only repeat the document's earlier model calls.
    LangChain wraps the SDK's ClientError, so the exception chain is searched.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ClientError):
            return current.code not in _RETRYABLE_CLIENT_CODES
        current = current.__cause__ or current.__context__
    return False


async def ai_request[T](request: Awaitable[T]) -> T:
    # Cancellation finishes before control returns to the persistent worker loop.
    async with asyncio.timeout(get_settings().ai_request_timeout_seconds):
        return await request


async def embed_texts(model: Embeddings, texts: list[str]) -> list[list[float]]:
    blank = [index for index, text in enumerate(texts) if not text.strip()]
    if blank:
        # Gemini answers 400 "content contains an empty Part" for the whole batch;
        # fail with the offending positions instead.
        raise ValueError(f"Cannot embed empty texts at positions {blank}")
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), _EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + _EMBEDDING_BATCH_SIZE]
        result = await ai_request(model.aembed_documents(batch))
        if len(result) != len(batch):
            raise ValueError("Embedding count does not match input count")
        embeddings.extend(result)
        logger.info("Embedded %d/%d texts", len(embeddings), len(texts))
    return embeddings
