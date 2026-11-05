import uuid
from dataclasses import replace

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import empty_checkpoint

from financial_assistant.agents.chat.graph import build_chat_graph
from financial_assistant.agents.chat.models import RetrievalGrade
from tests.utils import (
    FakeHybridRetriever,
    chat_context,
    chunk_source,
    community_source,
    fake_chat_models,
    tool_call_message,
)

pytestmark = pytest.mark.integration


def _config(thread_id: str):
    return {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}


async def test_setup_is_idempotent(langgraph_checkpointer):
    # migrated_database already ran services/setup_checkpointer once.
    await langgraph_checkpointer.setup()
    await langgraph_checkpointer.setup()


async def test_pool_returns_rows_by_column_name(langgraph_checkpointer):
    # The saver reads rows as dicts; the default tuple rows raise TypeError.
    async with langgraph_checkpointer.conn.connection() as connection:
        cursor = await connection.execute("SELECT 1 AS one")
        assert await cursor.fetchone() == {"one": 1}


async def test_delete_thread_removes_its_checkpoints_only(langgraph_checkpointer):
    deleted, kept = str(uuid.uuid4()), str(uuid.uuid4())
    for thread_id in (deleted, kept):
        await langgraph_checkpointer.aput(
            _config(thread_id),
            empty_checkpoint(),
            {"source": "input", "step": -1, "parents": {}},
            {},
        )
    try:
        await langgraph_checkpointer.adelete_thread(deleted)
        assert await langgraph_checkpointer.aget_tuple(_config(deleted)) is None
        assert await langgraph_checkpointer.aget_tuple(_config(kept)) is not None
    finally:
        await langgraph_checkpointer.adelete_thread(kept)


async def test_the_graph_persists_state_across_runs_until_the_thread_is_deleted(
    langgraph_checkpointer,
):
    models = fake_chat_models(
        [tool_call_message("revenue")],
        answer=[{"answer": "Revenue grew [1].", "cited_refs": [1]}],
    )
    retriever = FakeHybridRetriever([chunk_source(1), community_source(1)])
    graph = build_chat_graph(models, checkpointer=langgraph_checkpointer)
    first = chat_context(retriever)
    thread_id = str(first.thread_id)
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    try:
        for context, question in (
            (first, "First"),
            (replace(first, turn_id=uuid.uuid4()), "Second"),
        ):
            async for _ in graph.astream(
                {"question": question},
                config=config,
                context=context,
                stream_mode="custom",
                durability="exit",
            ):
                pass

        state = (await graph.aget_state(config)).values
        assert [message.content for message in state["messages"]] == [
            "First",
            "Revenue grew [1].",
            "Second",
            "Revenue grew [1].",
        ]
        # Our types survive the round trip through Postgres.
        cited = state["cited_sources"][str(first.turn_id)]
        assert cited == [{"ref": 1, "source": chunk_source(1)}]
        assert isinstance(state["grade"], RetrievalGrade)
    finally:
        await langgraph_checkpointer.adelete_thread(thread_id)
    assert (await graph.aget_state(config)).values == {}
