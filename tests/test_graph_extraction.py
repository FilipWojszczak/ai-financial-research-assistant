from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from financial_assistant.ai.graph_extraction import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
    _normalize_entity_type,
    process_document_graph,
)
from financial_assistant.models.document import ParentChunk
from financial_assistant.models.graph import EntityType

# ---------------------------------------------------------------------------
# _normalize_entity_type - pure unit tests, no I/O
# ---------------------------------------------------------------------------


def test_normalize_entity_type_known_types():
    assert _normalize_entity_type("COMPANY") == EntityType.COMPANY
    assert _normalize_entity_type("PERSON") == EntityType.PERSON
    assert _normalize_entity_type("FINANCIAL_METRIC") == EntityType.FINANCIAL_METRIC
    assert _normalize_entity_type("EVENT") == EntityType.EVENT
    assert _normalize_entity_type("PRODUCT") == EntityType.PRODUCT
    assert _normalize_entity_type("LOCATION") == EntityType.LOCATION


def test_normalize_entity_type_other_for_unknown():
    assert _normalize_entity_type("FOOBAR") == EntityType.OTHER
    assert _normalize_entity_type("") == EntityType.OTHER


def test_normalize_entity_type_case_insensitive():
    assert _normalize_entity_type("company") == EntityType.COMPANY
    assert _normalize_entity_type("Person") == EntityType.PERSON


# ---------------------------------------------------------------------------
# process_document_graph - mocked LLM and session
# ---------------------------------------------------------------------------


def _make_chunk(chunk_id: int, content: str = "Some text") -> ParentChunk:
    chunk = MagicMock(spec=ParentChunk)
    chunk.id = chunk_id
    chunk.content = content
    return chunk


def _make_extraction_result(
    entities: list[tuple[str, str]],
    relationships: list[tuple[str, str, str]],
) -> ExtractionResult:
    return ExtractionResult(
        entities=[
            ExtractedEntity(name=name, type=etype, description=None)
            for name, etype in entities
        ],
        relationships=[
            ExtractedRelationship(
                source=src, target=tgt, relationship_type=rel_type, description=None
            )
            for src, tgt, rel_type in relationships
        ],
    )


@pytest.mark.asyncio
async def test_process_document_graph_deduplicates_entities():
    """The same entity name across two chunks should produce only one Entity row."""
    chunk_a = _make_chunk(1, "Apple Inc. reported revenue.")
    chunk_b = _make_chunk(2, "Apple Inc. is headquartered in Cupertino.")

    extraction_a = _make_extraction_result(
        [("Apple Inc.", "COMPANY"), ("Tim Cook", "PERSON")],
        [("Tim Cook", "Apple Inc.", "CEO_OF")],
    )
    extraction_b = _make_extraction_result(
        [("Apple Inc.", "COMPANY"), ("Cupertino", "LOCATION")],
        [("Apple Inc.", "Cupertino", "OPERATES_IN")],
    )

    session = AsyncMock(spec=AsyncSession)
    session.flush = AsyncMock()
    added_objects: list = []
    session.add = lambda obj: added_objects.append(obj)

    with patch(
        "financial_assistant.ai.graph_extraction.extract_entities_and_relationships",
        new=AsyncMock(side_effect=[extraction_a, extraction_b]),
    ):
        entities = await process_document_graph(
            session=session,
            document_id=1,
            parent_chunks=[chunk_a, chunk_b],
        )

    entity_names = {e.name for e in entities}
    # "Apple Inc." appears in both chunks but should be stored once
    assert entity_names == {"Apple Inc.", "Tim Cook", "Cupertino"}
    assert len(entities) == 3


@pytest.mark.asyncio
async def test_process_document_graph_skips_self_relationships():
    """Relationships where source == target should be silently dropped."""
    chunk = _make_chunk(1)
    extraction = _make_extraction_result(
        [("Apple Inc.", "COMPANY")],
        [("Apple Inc.", "Apple Inc.", "SELF_REF")],
    )

    session = AsyncMock(spec=AsyncSession)
    session.flush = AsyncMock()
    added_objects: list = []
    session.add = lambda obj: added_objects.append(obj)

    with patch(
        "financial_assistant.ai.graph_extraction.extract_entities_and_relationships",
        new=AsyncMock(return_value=extraction),
    ):
        entities = await process_document_graph(
            session=session,
            document_id=1,
            parent_chunks=[chunk],
        )

    from financial_assistant.models.graph import EntityRelationship

    relationship_objects = [
        o for o in added_objects if isinstance(o, EntityRelationship)
    ]
    assert relationship_objects == []
    assert len(entities) == 1


@pytest.mark.asyncio
async def test_process_document_graph_skips_unknown_relationship_entities():
    """Relationships referencing entities not in the extraction should be dropped."""
    chunk = _make_chunk(1)
    extraction = _make_extraction_result(
        [("Apple Inc.", "COMPANY")],
        [("Apple Inc.", "Unknown Corp", "PARTNER_OF")],
    )

    session = AsyncMock(spec=AsyncSession)
    session.flush = AsyncMock()
    added_objects: list = []
    session.add = lambda obj: added_objects.append(obj)

    with patch(
        "financial_assistant.ai.graph_extraction.extract_entities_and_relationships",
        new=AsyncMock(return_value=extraction),
    ):
        await process_document_graph(
            session=session,
            document_id=1,
            parent_chunks=[chunk],
        )

    from financial_assistant.models.graph import EntityRelationship

    relationship_objects = [
        o for o in added_objects if isinstance(o, EntityRelationship)
    ]
    assert relationship_objects == []


@pytest.mark.asyncio
async def test_process_document_graph_tolerates_chunk_extraction_failure():
    """If a chunk's LLM call raises, processing continues with remaining chunks."""
    chunk_a = _make_chunk(1)
    chunk_b = _make_chunk(2)
    extraction_b = _make_extraction_result([("Microsoft", "COMPANY")], [])

    session = AsyncMock(spec=AsyncSession)
    session.flush = AsyncMock()
    session.add = MagicMock()

    with patch(
        "financial_assistant.ai.graph_extraction.extract_entities_and_relationships",
        new=AsyncMock(side_effect=[RuntimeError("LLM error"), extraction_b]),
    ):
        entities = await process_document_graph(
            session=session,
            document_id=1,
            parent_chunks=[chunk_a, chunk_b],
        )

    assert len(entities) == 1
    assert entities[0].name == "Microsoft"
