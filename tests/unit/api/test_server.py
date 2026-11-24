from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from financial_assistant.agents.chat.runner import ChatTurnRunner
from financial_assistant.api.server import lifespan
from tests.utils import fake_chat_models

pytestmark = pytest.mark.unit

_MODULE = "financial_assistant.api.server"
SHUTDOWN = [call.cancel(), call.trace(), call.close()]


@pytest.fixture
def shutdown():
    """Records the order of the shutdown steps."""
    order = MagicMock(cancel=AsyncMock(), close=AsyncMock())
    with (
        patch(f"{_MODULE}.init_langgraph_pool", new_callable=AsyncMock),
        patch(f"{_MODULE}.default_retriever"),
        patch(f"{_MODULE}.cancel_running_turns", order.cancel),
        patch(f"{_MODULE}.shutdown_tracing", order.trace),
        patch(f"{_MODULE}.close_langgraph_pool", order.close),
    ):
        yield order


async def test_lifespan_builds_the_runner_and_shuts_down_in_order(shutdown):
    app = FastAPI()
    with patch(f"{_MODULE}.create_chat_models", return_value=fake_chat_models([])):
        async with lifespan(app):
            assert isinstance(app.state.checkpointer, AsyncPostgresSaver)
            assert isinstance(app.state.chat_runner, ChatTurnRunner)
            assert app.state.chat_runner.graph.checkpointer is app.state.checkpointer
            assert shutdown.mock_calls == []
    assert shutdown.mock_calls == SHUTDOWN


async def test_a_failed_startup_still_closes_the_pool(shutdown):
    with (
        patch(f"{_MODULE}.create_chat_models", side_effect=ValueError("no API key")),
        pytest.raises(ValueError, match="no API key"),
    ):
        async with lifespan(FastAPI()):
            pass
    assert shutdown.mock_calls == SHUTDOWN
