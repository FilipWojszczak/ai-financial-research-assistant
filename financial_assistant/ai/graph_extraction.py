import logging
from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.checkpoint import CheckpointKind
from ..models.document import ParentChunk
from ..models.graph import Entity, EntityRelationship, EntityType
from .checkpoints import AttemptCheckpoints, checkpoint_key
from .requests import ai_request

logger = logging.getLogger(__name__)

_EXTRACTION_MODEL = "gemini-3.6-flash"
_extraction_llm = ChatGoogleGenerativeAI(
    model=_EXTRACTION_MODEL, temperature=0, max_retries=1
)


class ExtractedEntity(BaseModel):
    name: str = Field(description="The name of the entity as it appears in the text")
    type: Literal[
        "COMPANY", "PERSON", "FINANCIAL_METRIC", "EVENT", "PRODUCT", "LOCATION", "OTHER"
    ] = Field(description="The category of entity")
    description: str | None = Field(
        default=None,
        description="One sentence describing the entity's relevance in this context",
    )


class ExtractedRelationship(BaseModel):
    source: str = Field(description="Name of the source entity")
    target: str = Field(description="Name of the target entity")
    relationship_type: str = Field(
        description=(
            "Concise relationship type (e.g. CEO_OF, ACQUIRED, REPORTED, COMPETES_WITH)"
        )
    )
    description: str | None = Field(
        default=None, description="One sentence describing this relationship"
    )


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity] = Field(
        description="Named entities found in the text"
    )
    relationships: list[ExtractedRelationship] = Field(
        description="Relationships between the extracted entities"
    )


_structured_extractor = _extraction_llm.with_structured_output(ExtractionResult)

_EXTRACTION_PROMPT = """
You are a financial document analyst. Extract named entities and relationships from the text below.

Entity types:
- COMPANY: Organizations, companies, corporations, funds
- PERSON: Named individuals (executives, analysts, board members)
- FINANCIAL_METRIC: Specific figures or KPIs (revenue, EPS, net income, margins)
- EVENT: Business events (mergers, acquisitions, earnings releases, regulatory actions)
- PRODUCT: Products, services, brands, business segments
- LOCATION: Geographic locations relevant to business
- OTHER: Other important named entities

Only extract entities and relationships that are explicitly mentioned. Do not infer.

Text:
{text}
"""  # noqa: E501


def _normalize_entity_type(type_str: str) -> EntityType:
    _map = {
        "COMPANY": EntityType.COMPANY,
        "PERSON": EntityType.PERSON,
        "FINANCIAL_METRIC": EntityType.FINANCIAL_METRIC,
        "EVENT": EntityType.EVENT,
        "PRODUCT": EntityType.PRODUCT,
        "LOCATION": EntityType.LOCATION,
    }
    return _map.get(type_str.upper(), EntityType.OTHER)


async def extract_entities_and_relationships(chunk_text: str) -> ExtractionResult:
    """Call the LLM to extract entities and relationships from a text chunk."""
    result = await ai_request(
        _structured_extractor.ainvoke(_EXTRACTION_PROMPT.format(text=chunk_text))
    )
    return result  # type: ignore[return-value]


async def _extract_with_checkpoint(
    chunk_text: str, checkpoints: AttemptCheckpoints | None
) -> ExtractionResult:
    """Reuse the extraction from an earlier failed attempt of the same document."""
    if checkpoints is None:
        return await extract_entities_and_relationships(chunk_text)
    key = checkpoint_key(_EXTRACTION_MODEL, _EXTRACTION_PROMPT.format(text=chunk_text))
    cached = await checkpoints.get(CheckpointKind.GRAPH_EXTRACTION, key)
    if cached is not None:
        return ExtractionResult.model_validate(cached)
    result = await extract_entities_and_relationships(chunk_text)
    await checkpoints.put(
        CheckpointKind.GRAPH_EXTRACTION, key, result.model_dump(mode="json")
    )
    return result


async def process_document_graph(
    session: AsyncSession,
    document_id: int,
    parent_chunks: list[ParentChunk],
    checkpoints: AttemptCheckpoints | None = None,
) -> list[Entity]:
    """
    Extract entities and relationships from all parent chunks of a document,
    deduplicate entities by name, and persist everything to the database.
    Returns the saved Entity objects.
    """
    if not parent_chunks:
        raise ValueError("Cannot extract a graph without parent chunks")

    # entity_name_lower -> Entity (deduplication within a document)
    entity_map: dict[str, Entity] = {}
    # Collect relationship data until entity IDs are available
    pending_relationships: list[dict] = []
    for index, parent_chunk in enumerate(parent_chunks, start=1):
        # Provider errors must not produce a partially extracted COMPLETED document.
        result = await _extract_with_checkpoint(parent_chunk.content, checkpoints)
        logger.info(
            "Document %d: extracted graph chunk %d/%d",
            document_id,
            index,
            len(parent_chunks),
        )

        for extracted in result.entities:
            key = extracted.name.lower().strip()
            if key not in entity_map:
                entity = Entity(
                    name=extracted.name,
                    type=_normalize_entity_type(extracted.type),
                    description=extracted.description,
                    document_id=document_id,
                )
                session.add(entity)
                entity_map[key] = entity

        for rel in result.relationships:
            pending_relationships.append(
                {
                    "source_key": rel.source.lower().strip(),
                    "target_key": rel.target.lower().strip(),
                    "relationship_type": rel.relationship_type,
                    "description": rel.description,
                    "chunk_id": parent_chunk.id,
                }
            )

    # Flush so entities receive their DB-assigned IDs
    await session.flush()

    for rel_data in pending_relationships:
        source = entity_map.get(rel_data["source_key"])
        target = entity_map.get(rel_data["target_key"])
        if source is None or target is None or source.id == target.id:
            continue
        session.add(
            EntityRelationship(
                source_entity_id=source.id,
                target_entity_id=target.id,
                relationship_type=rel_data["relationship_type"],
                description=rel_data["description"],
                document_id=document_id,
                chunk_id=rel_data["chunk_id"],
            )
        )

    await session.flush()
    return list(entity_map.values())
