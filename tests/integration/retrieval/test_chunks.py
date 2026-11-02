import pytest
from sqlalchemy import text

from financial_assistant.models.document import DocumentStatus, DocumentType
from financial_assistant.retrieval.chunks import ChunkRetriever, bm25_ranking
from financial_assistant.retrieval.types import RetrievalScope
from tests.integration.retrieval.seeding import axis, retrieval_settings, vector

pytestmark = pytest.mark.integration

NO_MATCH = "zzzunmatchedterm"


async def _search(private_database, query, embedding, scope, **settings):
    retriever = ChunkRetriever(retrieval_settings(**settings))
    async with private_database.factory() as session:
        return await retriever.search(session, query, embedding, scope)


async def test_bm25_ranks_the_exact_term_chunk_first(private_database, seed):
    user = await seed.user("reader@example.com")
    document = await seed.document(
        [
            [("Headcount grew to 1,200 employees", axis(1))],
            [("EBITDA margin expanded to 31 percent", axis(2))],
            [("Capital expenditure was flat", axis(3))],
        ]
    )
    scope = RetrievalScope(user_id=user)
    async with private_database.factory() as session:
        bm25 = list(await session.execute(bm25_ranking("EBITDA margin", scope, 20)))
    assert [parent for _, parent in bm25] == [document.parent_ids[1]]

    # Dense distances all tie (orthogonal query), so BM25 decides the fused order.
    sources = await _search(private_database, "EBITDA margin", axis(10), scope)
    assert sources[0].parent_chunk_id == document.parent_ids[1]
    assert sources[0].content == "EBITDA margin expanded to 31 percent"


async def test_dense_ranking_finds_the_nearest_chunk(private_database, seed):
    user = await seed.user("reader@example.com")
    document = await seed.document(
        [[("Alpha", axis(1))], [("Beta", axis(2))], [("Gamma", vector(0, 1, 1))]]
    )
    # No BM25 hit: RRF runs over the dense list alone.
    sources = await _search(
        private_database, NO_MATCH, axis(2), RetrievalScope(user_id=user)
    )
    assert [source.parent_chunk_id for source in sources] == [
        document.parent_ids[1],
        document.parent_ids[2],
        document.parent_ids[0],
    ]
    assert sources[0].score == pytest.approx(1 / 61)


async def test_fusion_favours_a_chunk_ranked_in_both_lists(private_database, seed):
    user = await seed.user("reader@example.com")
    document = await seed.document(
        [
            [("Logistics overview", axis(0))],  # dense #1, no BM25 match
            [("Revenue grew 8 percent", vector(1, 1))],  # dense #2, BM25 match
            [("Unrelated text", axis(5))],
        ]
    )
    sources = await _search(
        private_database, "revenue", axis(0), RetrievalScope(user_id=user)
    )
    assert sources[0].parent_chunk_id == document.parent_ids[1]
    assert sources[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert sources[1].parent_chunk_id == document.parent_ids[0]


async def test_parents_are_deduplicated_ordered_by_best_child_and_capped(
    private_database, seed
):
    user = await seed.user("reader@example.com")
    document = await seed.document(
        [
            [("First child", axis(0)), ("Second child", vector(1, 0.1))],
            [("Middle", vector(1, 2))],
            [("Far", axis(3))],
        ]
    )
    scope = RetrievalScope(user_id=user)
    sources = await _search(private_database, NO_MATCH, axis(0), scope)
    assert [source.parent_chunk_id for source in sources] == document.parent_ids
    assert sources[0].content == "First child Second child"
    assert sources[0].score == pytest.approx(1 / 61)  # its best child's score

    capped = await _search(
        private_database, NO_MATCH, axis(0), scope, retrieval_max_parents=2
    )
    assert [source.parent_chunk_id for source in capped] == document.parent_ids[:2]


async def test_only_public_and_own_documents_are_searched(private_database, seed):
    reader = await seed.user("reader@example.com")
    other = await seed.user("other@example.com")
    deleted = await seed.user("deleted@example.com", tombstoned=True)
    chunk = [[("Revenue for the year", axis(0))]]
    public = await seed.document(chunk)
    own = await seed.document(chunk, owner_id=reader)
    others = await seed.document(chunk, owner_id=other)
    # A deleted account's documents keep its owner_id and match nobody.
    await seed.document(chunk, owner_id=deleted)

    for user, visible in (
        (reader, {public.id, own.id}),
        (other, {public.id, others.id}),
    ):
        sources = await _search(
            private_database, "revenue", axis(0), RetrievalScope(user_id=user)
        )
        assert {source.document_id for source in sources} == visible


@pytest.mark.parametrize("status", [DocumentStatus.PROCESSING, DocumentStatus.FAILED])
async def test_documents_that_are_not_completed_are_excluded(
    private_database, seed, status
):
    user = await seed.user("reader@example.com")
    completed = await seed.document([[("Revenue", axis(0))]])
    await seed.document([[("Revenue", axis(0))]], status=status)
    sources = await _search(
        private_database, "revenue", axis(0), RetrievalScope(user_id=user)
    )
    assert {source.document_id for source in sources} == {completed.id}


async def test_scope_filters_narrow_the_search(private_database, seed):
    user = await seed.user("reader@example.com")
    chunk = [[("Revenue", axis(0))]]
    nvda_10k = await seed.document(chunk, company_ticker="NVDA", year=2025)
    nvda_10q = await seed.document(
        chunk,
        company_ticker="NVDA",
        document_type=DocumentType.QUARTERLY_REPORT,
        year=2024,
    )
    tsla_10k = await seed.document(chunk, company_ticker="TSLA", year=2025)

    async def documents(**filters):
        sources = await _search(
            private_database,
            "revenue",
            axis(0),
            RetrievalScope(user_id=user, **filters),
        )
        return {source.document_id for source in sources}

    assert await documents(company_ticker="nvda") == {nvda_10k.id, nvda_10q.id}
    assert await documents(document_type=DocumentType.QUARTERLY_REPORT) == {nvda_10q.id}
    assert await documents(year=2025) == {nvda_10k.id, tsla_10k.id}
    assert await documents(document_ids=(tsla_10k.id,)) == {tsla_10k.id}
    assert await documents(document_ids=()) == set()


@pytest.mark.parametrize(
    "query",
    [
        """what's the "EPS" (diluted): AND -guidance*""",
        "revenue OR NOT ~fuzzy^2 [1 TO 5] {x} \\ / ? !",
        "   ",
    ],
)
async def test_query_syntax_characters_are_searched_as_plain_text(
    private_database, seed, query
):
    user = await seed.user("reader@example.com")
    document = await seed.document([[("Diluted EPS guidance", axis(0))]])
    sources = await _search(
        private_database, query, axis(0), RetrievalScope(user_id=user)
    )
    assert [source.document_id for source in sources] == [document.id]


async def test_the_hnsw_index_exists_in_the_migrated_schema(private_database):
    async with private_database.factory() as session:
        definition = await session.scalar(
            text(
                "SELECT indexdef FROM pg_indexes "
                "WHERE schemaname = 'public' AND indexname = 'ix_child_chunk_embedding'"
            )
        )
    assert definition is not None
    assert "USING hnsw (embedding vector_cosine_ops)" in definition
