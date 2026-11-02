"""Dense search over the GraphRAG community summaries."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import Settings
from ..models import Document, GraphCommunity
from .scope import accessible_documents
from .types import CommunitySource, RetrievalScope


class CommunityRetriever:
    name = "communities"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def search(
        self,
        session: AsyncSession,
        query: str,
        query_embedding: list[float],
        scope: RetrievalScope,
    ) -> list[CommunitySource]:
        settings = self._settings
        if settings.retrieval_community_k == 0:
            return []
        # ~300 rows and no vector index: an exact scan is fast and never misses rows.
        distance = GraphCommunity.embedding.cosine_distance(query_embedding)
        rows = await session.execute(
            select(
                GraphCommunity.id,
                GraphCommunity.title,
                GraphCommunity.summary,
                Document.id,
                Document.filename,
                Document.company_ticker,
                distance.label("distance"),
            )
            .join(Document, Document.id == GraphCommunity.document_id)
            .where(
                GraphCommunity.embedding.is_not(None),
                GraphCommunity.document_id.in_(accessible_documents(scope)),
            )
            .order_by(distance)
            .limit(settings.retrieval_community_k)
        )
        sources = []
        for community_id, title, summary, document_id, filename, ticker, d in rows:
            similarity = 1 - d
            # Dense search always returns k rows: keep only related summaries.
            if similarity >= settings.retrieval_community_min_similarity:
                sources.append(
                    CommunitySource(
                        community_id=community_id,
                        document_id=document_id,
                        filename=filename,
                        company_ticker=ticker,
                        title=title,
                        summary=summary,
                        similarity=similarity,
                    )
                )
        return sources
