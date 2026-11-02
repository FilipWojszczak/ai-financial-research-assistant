"""One search over every retriever with a single query embedding."""

from collections.abc import Sequence

from langchain_core.embeddings import Embeddings
from sqlalchemy.ext.asyncio import AsyncSession

from ..ai.embeddings import embeddings_model
from ..ai.requests import embed_query
from ..core.config import Settings, get_settings
from .chunks import ChunkRetriever
from .communities import CommunityRetriever
from .types import ChunkSource, CommunitySource, RetrievalScope, Retriever


class HybridRetriever:
    def __init__(self, retrievers: Sequence[Retriever], embeddings: Embeddings) -> None:
        self.retrievers = tuple(retrievers)
        self._embeddings = embeddings

    async def search(
        self, session: AsyncSession, query: str, scope: RetrievalScope
    ) -> list[ChunkSource | CommunitySource]:
        """Sources of all retrievers, in retriever order (chunks, then communities)."""
        if scope.is_empty:
            return []
        # Raises ValueError on a blank query. Embedded once for all retrievers.
        query_embedding = await embed_query(self._embeddings, query)
        sources: list[ChunkSource | CommunitySource] = []
        # Sequential: one AsyncSession can't run concurrent queries.
        for retriever in self.retrievers:
            sources.extend(
                await retriever.search(session, query, query_embedding, scope)
            )
        return sources


def default_retriever(settings: Settings | None = None) -> HybridRetriever:
    # An EntityRetriever (local graph search) would be appended here.
    settings = settings or get_settings()
    return HybridRetriever(
        [ChunkRetriever(settings), CommunityRetriever(settings)], embeddings_model
    )
