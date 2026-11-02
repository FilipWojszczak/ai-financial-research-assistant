"""Hybrid search over child chunks (BM25 + dense, fused with RRF) → parent chunks."""

from sqlalchemy import Select, Text, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import Settings
from ..models import ChildChunk, Document, ParentChunk
from .fusion import reciprocal_rank_fusion
from .scope import accessible_documents
from .types import ChunkSource, RetrievalScope


def bm25_ranking(query: str, scope: RetrievalScope, k: int) -> Select[tuple[int, int]]:
    """(child id, parent id) of the best BM25 matches in the scope."""
    return (
        select(ChildChunk.id, ChildChunk.parent_id)
        .join(ParentChunk, ParentChunk.id == ChildChunk.parent_id)
        .where(
            # ||| (match disjunction) tokenizes the text with the field's tokenizer
            # and parses no query syntax, so quotes, AND, -, * or : can't break it.
            # The cast picks the text overload of the operator.
            ChildChunk.content.op("|||")(cast(query, Text)),
            ParentChunk.document_id.in_(accessible_documents(scope)),
        )
        .order_by(func.pdb.score(ChildChunk.id).desc())
        .limit(k)
    )


def dense_ranking(
    query_embedding: list[float], scope: RetrievalScope, k: int
) -> Select[tuple[int, int]]:
    """(child id, parent id) of the nearest child chunks in the scope."""
    # <=> matches the vector_cosine_ops HNSW index ix_child_chunk_embedding.
    distance = ChildChunk.embedding.cosine_distance(query_embedding)
    nearest = (
        select(ChildChunk.id, ChildChunk.parent_id, distance.label("distance"))
        .join(ParentChunk, ParentChunk.id == ChildChunk.parent_id)
        .where(ParentChunk.document_id.in_(accessible_documents(scope)))
        .order_by(distance)
        .limit(k)
        .subquery()
    )
    # relaxed_order may return the index scan slightly out of order: re-sort.
    return select(nearest.c.id, nearest.c.parent_id).order_by(nearest.c.distance)


async def configure_dense_search(session: AsyncSession, ef_search: int) -> None:
    """
    Transaction-local HNSW settings. A filter on the joined documents can leave a
    plain HNSW scan with fewer than k rows; iterative scan keeps searching.
    """
    await session.execute(
        select(
            func.set_config("hnsw.ef_search", str(ef_search), True),
            func.set_config("hnsw.iterative_scan", "relaxed_order", True),
        )
    )


class ChunkRetriever:
    name = "chunks"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def search(
        self,
        session: AsyncSession,
        query: str,
        query_embedding: list[float],
        scope: RetrievalScope,
    ) -> list[ChunkSource]:
        settings = self._settings
        bm25: list[tuple[int, int]] = []
        if query.strip():
            bm25 = await _ranking(
                session, bm25_ranking(query, scope, settings.retrieval_bm25_k)
            )
        await configure_dense_search(session, settings.retrieval_hnsw_ef_search)
        dense = await _ranking(
            session, dense_ranking(query_embedding, scope, settings.retrieval_dense_k)
        )

        parent_of = dict(bm25) | dict(dense)
        fused = reciprocal_rank_fusion(
            [[child for child, _ in bm25], [child for child, _ in dense]],
            k=settings.retrieval_rrf_k,
        )[: settings.retrieval_fused_children]
        # Parents in the order of their best child; that child's score is theirs.
        parent_scores: dict[int, float] = {}
        for child, score in fused:
            parent_scores.setdefault(parent_of[child], score)
            if len(parent_scores) == settings.retrieval_max_parents:
                break
        if not parent_scores:
            return []

        # Parent content and document metadata in one query.
        rows = await session.execute(
            select(
                ParentChunk.id,
                ParentChunk.content,
                ParentChunk.document_id,
                Document.filename,
                Document.company_ticker,
                Document.document_type,
                Document.year,
            )
            .join(Document, Document.id == ParentChunk.document_id)
            .where(ParentChunk.id.in_(parent_scores))
        )
        sources = {
            row.id: ChunkSource(
                parent_chunk_id=row.id,
                document_id=row.document_id,
                filename=row.filename,
                company_ticker=row.company_ticker,
                document_type=row.document_type,
                year=row.year,
                content=row.content,
                score=parent_scores[row.id],
            )
            for row in rows
        }
        return [sources[parent] for parent in parent_scores if parent in sources]


async def _ranking(
    session: AsyncSession, query: Select[tuple[int, int]]
) -> list[tuple[int, int]]:
    return [(child, parent) for child, parent in await session.execute(query)]
