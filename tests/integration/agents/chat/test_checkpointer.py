import uuid

import pytest
from langgraph.checkpoint.base import empty_checkpoint

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
