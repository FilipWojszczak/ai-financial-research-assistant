from typing import cast
from unittest.mock import MagicMock

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy.ext.asyncio import AsyncSession

from financial_assistant.models.document import DocumentType
from financial_assistant.retrieval.hybrid import HybridRetriever
from financial_assistant.retrieval.types import (
    ChunkSource,
    CommunitySource,
    RetrievalScope,
)

pytestmark = pytest.mark.unit

SCOPE = RetrievalScope(user_id=1)
CHUNK = ChunkSource(
    parent_chunk_id=1,
    document_id=1,
    filename="a.pdf",
    company_ticker="A",
    document_type=DocumentType.ANNUAL_REPORT,
    year=2025,
    content="text",
    score=0.03,
)
COMMUNITY = CommunitySource(
    community_id=1,
    document_id=1,
    filename="a.pdf",
    company_ticker="A",
    title="t",
    summary="s",
    similarity=0.9,
)


class CountingEmbedding(DeterministicFakeEmbedding):
    calls: int = 0

    async def aembed_query(self, text: str) -> list[float]:
        self.calls += 1
        return await super().aembed_query(text)


class FakeRetriever:
    def __init__(self, name, sources, calls) -> None:
        self.name = name
        self._sources = sources
        self._calls = calls

    async def search(self, session, query, query_embedding, scope):
        self._calls.append((self.name, query, query_embedding, scope))
        return self._sources


async def test_query_is_embedded_once_and_retrievers_run_in_order():
    embeddings = CountingEmbedding(size=768)
    calls = []
    hybrid = HybridRetriever(
        [
            FakeRetriever("chunks", [CHUNK], calls),
            FakeRetriever("communities", [COMMUNITY], calls),
        ],
        embeddings,
    )
    session = cast(AsyncSession, MagicMock())
    sources = await hybrid.search(session, "data center revenue", SCOPE)
    assert sources == [CHUNK, COMMUNITY]
    assert embeddings.calls == 1
    assert [name for name, *_ in calls] == ["chunks", "communities"]
    expected = await embeddings.aembed_query("data center revenue")
    for _, query, query_embedding, scope in calls:
        assert query == "data center revenue"
        assert query_embedding == expected
        assert scope is SCOPE


async def test_blank_query_is_rejected_before_any_search():
    calls = []
    hybrid = HybridRetriever(
        [FakeRetriever("chunks", [CHUNK], calls)], DeterministicFakeEmbedding(size=768)
    )
    with pytest.raises(ValueError, match="empty query"):
        await hybrid.search(cast(AsyncSession, MagicMock()), "   ", SCOPE)
    assert calls == []


async def test_empty_scope_returns_nothing_without_embedding():
    embeddings = CountingEmbedding(size=768)
    calls = []
    hybrid = HybridRetriever([FakeRetriever("chunks", [CHUNK], calls)], embeddings)
    empty = RetrievalScope(user_id=1, document_ids=())
    assert await hybrid.search(cast(AsyncSession, MagicMock()), "revenue", empty) == []
    assert embeddings.calls == 0
    assert calls == []
