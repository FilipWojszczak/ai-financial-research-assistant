from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from financial_assistant.ai.community_detection import (
    _generate_community_summary,
    process_document_communities,
)
from financial_assistant.models.graph import (
    Entity,
    EntityRelationship,
    GraphCommunity,
    GraphCommunityMembership,
)


def _make_entity(entity_id: int, name: str = "Entity") -> Entity:
    entity = MagicMock(spec=Entity)
    entity.id = entity_id
    entity.name = name
    entity.type = "company"
    entity.description = None
    return entity


def _make_relationship(
    source_id: int, target_id: int, rel_type: str = "RELATED_TO"
) -> EntityRelationship:
    rel = MagicMock(spec=EntityRelationship)
    rel.source_entity_id = source_id
    rel.target_entity_id = target_id
    rel.relationship_type = rel_type
    rel.description = None
    return rel


def _make_session(relationships: list) -> tuple[AsyncMock, list]:
    session = AsyncMock(spec=AsyncSession)
    session.flush = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = relationships
    session.execute.return_value = mock_result
    added_objects: list = []
    session.add = lambda obj: added_objects.append(obj)
    return session, added_objects


# ---------------------------------------------------------------------------
# _generate_community_summary - LLM response parsing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_community_summary_parses_title_and_summary():
    """Correctly structured LLM response is split into title and summary."""
    mock_response = MagicMock()
    mock_response.content = (
        "TITLE: Apple Leadership\nSUMMARY: Tim Cook leads Apple as CEO."
    )

    with patch("financial_assistant.ai.community_detection._summary_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        title, summary = await _generate_community_summary(
            [_make_entity(1, "Apple"), _make_entity(2, "Tim Cook")], []
        )

    assert title == "Apple Leadership"
    assert summary == "Tim Cook leads Apple as CEO."


@pytest.mark.asyncio
async def test_generate_community_summary_falls_back_when_format_not_followed():
    """When the LLM ignores the TITLE:/SUMMARY: format, safe defaults are returned."""
    mock_response = MagicMock()
    mock_response.content = "Some unstructured response"

    with patch("financial_assistant.ai.community_detection._summary_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        title, summary = await _generate_community_summary([_make_entity(1)], [])

    assert title == "Community"
    assert summary == "Some unstructured response"


# ---------------------------------------------------------------------------
# process_document_communities - mocked LLM, embeddings, and session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_document_communities_returns_early_when_no_entities():
    """Empty entity list skips all processing including the DB query."""
    session = AsyncMock(spec=AsyncSession)
    await process_document_communities(session, document_id=1, entities=[])
    session.execute.assert_not_called()


@pytest.mark.asyncio
async def test_process_document_communities_returns_early_when_no_edges():
    """Entities with no relationships produce no graph edges and nothing is saved."""
    entities = [_make_entity(1, "Apple"), _make_entity(2, "Microsoft")]
    session, added_objects = _make_session(relationships=[])

    await process_document_communities(session, document_id=1, entities=entities)

    assert added_objects == []


@pytest.mark.asyncio
async def test_process_document_communities_saves_communities_and_memberships():
    """Two disconnected entity pairs produce two communities with correct memberships."""  # noqa: E501
    entities = [
        _make_entity(1, "Apple"),
        _make_entity(2, "Tim Cook"),
        _make_entity(3, "Microsoft"),
        _make_entity(4, "Satya Nadella"),
    ]
    relationships = [
        _make_relationship(1, 2, "CEO_OF"),
        _make_relationship(3, 4, "CEO_OF"),
    ]
    session, added_objects = _make_session(relationships)

    with (
        patch(
            "financial_assistant.ai.community_detection._generate_community_summary",
            new=AsyncMock(return_value=("Test Title", "Test summary.")),
        ),
        patch(
            "financial_assistant.ai.community_detection._embeddings_model"
        ) as mock_embed,
    ):
        mock_embed.aembed_documents = AsyncMock(return_value=[[0.1] * 768, [0.2] * 768])
        await process_document_communities(session, document_id=1, entities=entities)

    communities = [o for o in added_objects if isinstance(o, GraphCommunity)]
    memberships = [o for o in added_objects if isinstance(o, GraphCommunityMembership)]

    assert len(communities) == 2
    assert len(memberships) == 4
    assert all(c.title == "Test Title" for c in communities)


@pytest.mark.asyncio
async def test_process_document_communities_skips_singleton_communities():
    """An isolated entity forms a singleton community that is silently skipped."""
    entities = [
        _make_entity(1, "Apple"),
        _make_entity(2, "Tim Cook"),
        _make_entity(3, "Google"),  # isolated — no relationships
    ]
    relationships = [_make_relationship(1, 2, "CEO_OF")]
    session, added_objects = _make_session(relationships)

    with (
        patch(
            "financial_assistant.ai.community_detection._generate_community_summary",
            new=AsyncMock(return_value=("Title", "Summary")),
        ),
        patch(
            "financial_assistant.ai.community_detection._embeddings_model"
        ) as mock_embed,
    ):
        mock_embed.aembed_documents = AsyncMock(return_value=[[0.1] * 768])
        await process_document_communities(session, document_id=1, entities=entities)

    communities = [o for o in added_objects if isinstance(o, GraphCommunity)]
    memberships = [o for o in added_objects if isinstance(o, GraphCommunityMembership)]

    assert len(communities) == 1
    assert len(memberships) == 2


@pytest.mark.asyncio
async def test_process_document_communities_uses_fallback_summary_on_llm_failure():
    """LLM failure during summarisation falls back to a generic title and summary."""
    entities = [_make_entity(1, "Apple"), _make_entity(2, "Tim Cook")]
    relationships = [_make_relationship(1, 2, "CEO_OF")]
    session, added_objects = _make_session(relationships)

    with (
        patch(
            "financial_assistant.ai.community_detection._generate_community_summary",
            new=AsyncMock(side_effect=RuntimeError("LLM unavailable")),
        ),
        patch(
            "financial_assistant.ai.community_detection._embeddings_model"
        ) as mock_embed,
    ):
        mock_embed.aembed_documents = AsyncMock(return_value=[[0.1] * 768])
        await process_document_communities(session, document_id=1, entities=entities)

    communities = [o for o in added_objects if isinstance(o, GraphCommunity)]
    assert len(communities) == 1
    assert communities[0].title == "Community of 2 entities"
    assert communities[0].summary == "A cluster of 2 related financial entities."


@pytest.mark.asyncio
async def test_process_document_communities_uses_none_embedding_on_embed_failure():
    """Embedding API failure stores None and still persists the community."""
    entities = [_make_entity(1, "Apple"), _make_entity(2, "Tim Cook")]
    relationships = [_make_relationship(1, 2, "CEO_OF")]
    session, added_objects = _make_session(relationships)

    with (
        patch(
            "financial_assistant.ai.community_detection._generate_community_summary",
            new=AsyncMock(return_value=("Title", "Summary")),
        ),
        patch(
            "financial_assistant.ai.community_detection._embeddings_model"
        ) as mock_embed,
    ):
        mock_embed.aembed_documents = AsyncMock(side_effect=RuntimeError("embed error"))
        await process_document_communities(session, document_id=1, entities=entities)

    communities = [o for o in added_objects if isinstance(o, GraphCommunity)]
    assert len(communities) == 1
    assert communities[0].embedding is None


@pytest.mark.asyncio
async def test_process_document_communities_falls_back_to_greedy_on_louvain_failure():
    """If Louvain raises, community detection continues with greedy modularity."""
    entities = [_make_entity(1, "Apple"), _make_entity(2, "Tim Cook")]
    relationships = [_make_relationship(1, 2, "CEO_OF")]
    session, added_objects = _make_session(relationships)

    with (
        patch(
            "financial_assistant.ai.community_detection.nx.community.louvain_communities",
            side_effect=RuntimeError("Louvain failed"),
        ),
        patch(
            "financial_assistant.ai.community_detection._generate_community_summary",
            new=AsyncMock(return_value=("Title", "Summary")),
        ),
        patch(
            "financial_assistant.ai.community_detection._embeddings_model"
        ) as mock_embed,
    ):
        mock_embed.aembed_documents = AsyncMock(return_value=[[0.1] * 768])
        await process_document_communities(session, document_id=1, entities=entities)

    communities = [o for o in added_objects if isinstance(o, GraphCommunity)]
    assert len(communities) == 1
