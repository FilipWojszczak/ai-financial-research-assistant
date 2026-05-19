import logging

import networkx as nx
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.graph import (
    Entity,
    EntityRelationship,
    GraphCommunity,
    GraphCommunityMembership,
)

logger = logging.getLogger(__name__)

_summary_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
_embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", output_dimensionality=768
)

_SUMMARY_PROMPT = """
You are analysing a cluster of related entities from a financial document.

Entities:
{entities}

Key relationships:
{relationships}

Write a concise title (5-10 words) and a 2-3 sentence summary that captures what connects these entities and why they matter.

Format:
TITLE: <title>
SUMMARY: <summary>
"""  # noqa: E501


async def _generate_community_summary(
    entities: list[Entity],
    relationships: list[EntityRelationship],
) -> tuple[str, str]:
    entity_lines = "\n".join(
        f"- {e.name} ({e.type}): {e.description or 'no description'}" for e in entities
    )
    rel_lines = (
        "\n".join(
            f"- {r.relationship_type}: {r.description or ''}"
            for r in relationships[:10]  # cap to avoid prompt blowout
        )
        or "None identified"
    )

    response = await _summary_llm.ainvoke(
        _SUMMARY_PROMPT.format(entities=entity_lines, relationships=rel_lines)
    )
    content = str(response.content).strip()

    title = "Community"
    summary = content
    for line in content.splitlines():
        if line.startswith("TITLE:"):
            title = line[6:].strip()
        elif line.startswith("SUMMARY:"):
            summary = line[8:].strip()

    return title, summary


async def process_document_communities(
    session: AsyncSession,
    document_id: int,
    entities: list[Entity],
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
        logger.warning("Louvain failed, falling back to greedy modularity communities")
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

        try:
            title, summary = await _generate_community_summary(
                community_entities, community_rels
            )
        except Exception:
            logger.warning(
                "Failed to summarise a community for document %d", document_id
            )
            n = len(community_entities)
            title = f"Community of {n} entities"
            summary = f"A cluster of {n} related financial entities."

        community_data.append((community_entities, title, summary))

    if not community_data:
        return

    # Batch-embed all summaries at once
    summaries = [summary for _, _, summary in community_data]
    try:
        embeddings = await _embeddings_model.aembed_documents(summaries)
    except Exception:
        logger.warning(
            "Failed to embed community summaries for document %d", document_id
        )
        embeddings = [None] * len(summaries)

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
