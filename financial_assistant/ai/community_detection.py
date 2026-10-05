import logging
from functools import partial

import networkx as nx
from langchain_core.exceptions import OutputParserException
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.checkpoint import CheckpointKind
from ..models.graph import (
    Entity,
    EntityRelationship,
    GraphCommunity,
    GraphCommunityMembership,
)
from .checkpoints import AttemptCheckpoints, checkpoint_key
from .requests import ai_request, embed_texts

logger = logging.getLogger(__name__)

_SUMMARY_MODEL = "gemini-3.6-flash"
_summary_llm = ChatGoogleGenerativeAI(
    model=_SUMMARY_MODEL, temperature=0, max_retries=1
)
_embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", output_dimensionality=768
)


class CommunitySummary(BaseModel):
    title: str = Field(description="Concise title of 5-10 words")
    summary: str = Field(
        description=(
            "2-3 sentences on what connects these entities and why they matter"
        )
    )


_structured_summarizer = _summary_llm.with_structured_output(CommunitySummary)

_SUMMARY_PROMPT = """
You are analysing a cluster of related entities from a financial document.

Entities:
{entities}

Key relationships:
{relationships}

Write a concise title (5-10 words) and a 2-3 sentence summary that captures what connects these entities and why they matter.
"""  # noqa: E501


async def _generate_community_summary(
    entities: list[Entity],
    relationships: list[EntityRelationship],
    checkpoints: AttemptCheckpoints | None = None,
) -> tuple[str, str]:
    # Sorted, so a retry builds the same prompt (and checkpoint key) even though
    # entity IDs and set iteration order differ between attempts.
    entity_lines = "\n".join(
        sorted(
            f"- {e.name} ({e.type}): {e.description or 'no description'}"
            for e in entities
        )
    )
    ordered_relationships = sorted(
        relationships, key=lambda r: (r.relationship_type, r.description or "")
    )
    rel_lines = (
        "\n".join(
            f"- {r.relationship_type}: {r.description or ''}"
            for r in ordered_relationships[:10]  # cap to avoid prompt blowout
        )
        or "None identified"
    )
    prompt = _SUMMARY_PROMPT.format(entities=entity_lines, relationships=rel_lines)

    key = checkpoint_key(_SUMMARY_MODEL, prompt)
    if checkpoints is not None:
        cached = await checkpoints.get(CheckpointKind.COMMUNITY_SUMMARY, key)
        if cached is not None:
            return cached["title"], cached["summary"]

    try:
        response: CommunitySummary = await ai_request(  # type: ignore[assignment]
            partial(_structured_summarizer.ainvoke, prompt)
        )
        title, summary = response.title.strip(), response.summary.strip()
    except OutputParserException:
        # Empty or truncated output (e.g. a safety block). At temperature 0 the same
        # prompt fails again, so a task retry would only dead-letter the document.
        logger.warning("Community summary response was not parseable", exc_info=True)
        title, summary = "", ""
    title = title or "Community"
    if not summary:
        # The summary is embedded next, and the embedding API rejects empty input.
        # The schema cannot forbid blank strings, so fall back deterministically.
        logger.warning(
            "Community summary response had no usable summary; using entity fallback"
        )
        summary = _fallback_summary(entities, ordered_relationships)

    if checkpoints is not None:
        await checkpoints.put(
            CheckpointKind.COMMUNITY_SUMMARY, key, {"title": title, "summary": summary}
        )
    return title, summary


def _fallback_summary(
    entities: list[Entity], relationships: list[EntityRelationship]
) -> str:
    names = ", ".join(sorted(e.name for e in entities)[:10])
    relationship_types = ", ".join(
        sorted({r.relationship_type for r in relationships})[:5]
    )
    if relationship_types:
        return f"Related entities: {names}. Relationships: {relationship_types}."
    return f"Related entities: {names}."


async def process_document_communities(
    session: AsyncSession,
    document_id: int,
    entities: list[Entity],
    checkpoints: AttemptCheckpoints | None = None,
) -> None:
    """
    Build a graph of entity relationships, detect communities with the Louvain
    algorithm, generate a summary for each community, embed the summary, and
    persist everything to the database.
    """
    if not entities:
        return

    entity_by_id = {e.id: e for e in entities}

    # Build undirected graph
    graph: nx.Graph = nx.Graph()
    for entity in entities:
        graph.add_node(entity.id)

    result = await session.execute(
        select(EntityRelationship).where(EntityRelationship.document_id == document_id)
    )
    all_relationships = result.scalars().all()

    for rel in all_relationships:
        src_known = rel.source_entity_id in entity_by_id
        tgt_known = rel.target_entity_id in entity_by_id
        if src_known and tgt_known:
            if graph.has_edge(rel.source_entity_id, rel.target_entity_id):
                graph[rel.source_entity_id][rel.target_entity_id]["weight"] += 1.0
            else:
                graph.add_edge(rel.source_entity_id, rel.target_entity_id, weight=1.0)

    if graph.number_of_edges() == 0:
        logger.info(
            "Document %d has no entity relationships; skipping community detection",
            document_id,
        )
        return

    # Community detection
    try:
        community_sets = nx.community.louvain_communities(graph, seed=42)
    except Exception:
        logger.warning(
            "Louvain failed, falling back to greedy modularity communities",
            exc_info=True,
        )
        community_sets = list(nx.community.greedy_modularity_communities(graph))

    rel_lookup: dict[frozenset[int], list[EntityRelationship]] = {}
    for rel in all_relationships:
        key: frozenset[int] = frozenset({rel.source_entity_id, rel.target_entity_id})
        rel_lookup.setdefault(key, []).append(rel)

    # Collect community summaries so we can batch-embed them
    community_data: list[tuple[list[Entity], str, str]] = []
    for community_set in community_sets:
        if len(community_set) < 2:
            continue

        community_entities = [
            entity_by_id[eid] for eid in community_set if eid in entity_by_id
        ]
        member_ids = set(community_set)
        community_rels = [
            rel for key, rels in rel_lookup.items() if key <= member_ids for rel in rels
        ]

        title, summary = await _generate_community_summary(
            community_entities, community_rels, checkpoints
        )
        logger.info(
            "Document %d: summarized community %d", document_id, len(community_data) + 1
        )

        community_data.append((community_entities, title, summary))

    if not community_data:
        return

    # Embed summaries in bounded batches
    summaries = [summary for _, _, summary in community_data]
    embeddings = await embed_texts(_embeddings_model, summaries)

    for (community_entities, title, summary), embedding in zip(
        community_data, embeddings, strict=True
    ):
        community = GraphCommunity(
            document_id=document_id,
            level=0,
            title=title,
            summary=summary,
            embedding=embedding,
        )
        session.add(community)
        await session.flush()

        for entity in community_entities:
            session.add(
                GraphCommunityMembership(
                    community_id=community.id,
                    entity_id=entity.id,
                )
            )

    await session.flush()
