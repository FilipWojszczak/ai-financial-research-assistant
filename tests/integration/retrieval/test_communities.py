import pytest

from financial_assistant.retrieval.communities import CommunityRetriever
from financial_assistant.retrieval.types import RetrievalScope
from tests.integration.retrieval.seeding import axis, retrieval_settings, vector

pytestmark = pytest.mark.integration


async def _search(private_database, scope, **settings):
    retriever = CommunityRetriever(retrieval_settings(**settings))
    async with private_database.factory() as session:
        return await retriever.search(session, "query", axis(0), scope)


async def test_similarity_floor_drops_unrelated_summaries(private_database, seed):
    user = await seed.user("reader@example.com")
    document = await seed.document([[("text", axis(0))]])
    close = await seed.community(document.id, "Close", axis(0))
    related = await seed.community(document.id, "Related", vector(1, 1))  # 0.71
    await seed.community(document.id, "Unrelated", axis(1))  # similarity 0
    await seed.community(document.id, "Not embedded", None)

    sources = await _search(private_database, RetrievalScope(user_id=user))
    assert [source.community_id for source in sources] == [close, related]
    assert sources[0].similarity == pytest.approx(1.0)
    assert sources[1].similarity == pytest.approx(0.5**0.5)
    assert sources[0].title == "Close"
    assert sources[0].summary == "Summary of Close"

    strict = await _search(
        private_database,
        RetrievalScope(user_id=user),
        retrieval_community_min_similarity=0.9,
    )
    assert [source.community_id for source in strict] == [close]


async def test_top_k_limits_the_results(private_database, seed):
    user = await seed.user("reader@example.com")
    document = await seed.document([[("text", axis(0))]])
    best = await seed.community(document.id, "Best", axis(0))
    await seed.community(document.id, "Second", vector(1, 0.2))
    scope = RetrievalScope(user_id=user)
    sources = await _search(private_database, scope, retrieval_community_k=1)
    assert [source.community_id for source in sources] == [best]
    assert await _search(private_database, scope, retrieval_community_k=0) == []


async def test_only_communities_of_accessible_documents_are_searched(
    private_database, seed
):
    reader = await seed.user("reader@example.com")
    other = await seed.user("other@example.com")
    public = await seed.document([[("text", axis(0))]])
    private = await seed.document([[("text", axis(0))]], owner_id=other)
    visible = await seed.community(public.id, "Public", axis(0))
    await seed.community(private.id, "Private", axis(0))

    sources = await _search(private_database, RetrievalScope(user_id=reader))
    assert [source.community_id for source in sources] == [visible]
    assert sources[0].document_id == public.id
