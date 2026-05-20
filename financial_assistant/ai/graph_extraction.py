import logging
from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.document import ParentChunk
from ..models.graph import Entity, EntityRelationship, EntityType

logger = logging.getLogger(__name__)

_extraction_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)


class ExtractedEntity(BaseModel):
    name: str = Field(description="The name of the entity as it appears in the text")
    type: Literal[
        "COMPANY", "PERSON", "FINANCIAL_METRIC", "EVENT", "PRODUCT", "LOCATION", "OTHER"
    ] = Field(description="The category of entity")
    description: str | None = Field(
        None,
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
        None, description="One sentence describing this relationship"
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
    result = await _structured_extractor.ainvoke(
        _EXTRACTION_PROMPT.format(text=chunk_text)
    )
    return result  # type: ignore[return-value]


async def process_document_graph(
    session: AsyncSession,
    document_id: int,
    parent_chunks: list[ParentChunk],
) -> list[Entity]:
    """
    Extract entities and relationships from all parent chunks of a document,
    deduplicate entities by name, and persist everything to the database.
    Returns the saved Entity objects.
    """
    # entity_name_lower -> Entity (deduplication within a document)
    entity_map: dict[str, Entity] = {}
    # Collect relationship data until entity IDs are available
    pending_relationships: list[dict] = []

    for parent_chunk in parent_chunks:
        try:
            result = await extract_entities_and_relationships(parent_chunk.content)
        except Exception:
            logger.warning(
                "Failed to extract entities from chunk %d, skipping", parent_chunk.id
            )
            continue

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
