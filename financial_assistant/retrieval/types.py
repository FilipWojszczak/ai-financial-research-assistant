"""What a search may see (RetrievalScope) and what it returns (Source)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.document import DocumentType


@dataclass(frozen=True)
class RetrievalScope:
    """
    Documents a search may use. Access control (the user) is always applied by
    retrieval/scope.py; the other fields only narrow it further.
    """

    user_id: int
    # Thread filters, fixed when the thread is created. () matches no document.
    document_ids: tuple[int, ...] | None = None
    company_ticker: str | None = None
    # Narrowing requested by the agent for one search.
    document_type: DocumentType | None = None
    year: int | None = None

    @property
    def is_empty(self) -> bool:
        return self.document_ids == ()

    def narrowed(
        self,
        *,
        company_ticker: str | None = None,
        document_type: DocumentType | None = None,
        year: int | None = None,
    ) -> RetrievalScope:
        """
        Intersect with the requested filters, never replace the existing ones: a
        thread scoped to NVDA asked for TSLA yields an empty scope, not a TSLA one.
        """
        if (
            _conflicts(self.company_ticker, company_ticker, str.upper)
            or _conflicts(self.document_type, document_type)
            or _conflicts(self.year, year)
        ):
            return replace(self, document_ids=())
        return replace(
            self,
            company_ticker=self.company_ticker or company_ticker,
            document_type=self.document_type or document_type,
            year=self.year if self.year is not None else year,
        )


def _conflicts[T](current: T | None, requested: T | None, key=None) -> bool:
    if current is None or requested is None:
        return False
    if key is not None:
        return key(current) != key(requested)
    return current != requested


class ChunkSource(BaseModel):
    kind: Literal["chunk"] = "chunk"
    parent_chunk_id: int
    document_id: int
    filename: str
    company_ticker: str
    document_type: DocumentType
    year: int
    # The parent chunk: the context around the matching child chunks.
    content: str
    # RRF score of its best child chunk.
    score: float


class CommunitySource(BaseModel):
    kind: Literal["community"] = "community"
    community_id: int
    document_id: int
    filename: str
    company_ticker: str
    title: str
    summary: str
    # 1 - cosine distance to the query.
    similarity: float


# Later: EntitySource, WebSource.
Source = Annotated[ChunkSource | CommunitySource, Field(discriminator="kind")]


def source_key(source: ChunkSource | CommunitySource) -> tuple[str, int]:
    """Identity of a source, for deduplication across searches."""
    if isinstance(source, ChunkSource):
        return (source.kind, source.parent_chunk_id)
    return (source.kind, source.community_id)


class Retriever(Protocol):
    name: str

    async def search(
        self,
        session: AsyncSession,
        query: str,
        query_embedding: list[float],
        scope: RetrievalScope,
    ) -> Sequence[ChunkSource | CommunitySource]: ...
