"""Upload → ingestion by a real worker → a chat answer over real retrieval."""

import json
import os

import pytest
from langchain_core.embeddings import Embeddings
from langgraph.checkpoint.memory import InMemorySaver

from financial_assistant.agents.chat.graph import build_chat_graph
from financial_assistant.agents.chat.runner import ChatTurnRunner
from financial_assistant.agents.chat.state import chat_serializer
from financial_assistant.api.dependencies.chat import get_chat_runner
from financial_assistant.api.server import app
from financial_assistant.core.config import get_settings
from financial_assistant.retrieval.chunks import ChunkRetriever
from financial_assistant.retrieval.communities import CommunityRetriever
from financial_assistant.retrieval.hybrid import HybridRetriever
from tests.utils import fake_chat_models, tool_call_message

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.getenv("RUN_INGESTION_TESTS") != "1",
        reason="Requires RUN_INGESTION_TESTS=1, PostgreSQL, and RabbitMQ management",
    ),
]


class ConstantEmbeddings(Embeddings):
    """The vector tests/e2e/ingestion_worker.py stores for every text."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 768


async def test_a_chat_answer_cites_the_ingested_document(flow):
    settings = get_settings()
    models = fake_chat_models(
        [tool_call_message("Acme revenue")],
        answer=[{"answer": "Acme reports revenue [1].", "cited_refs": [1]}],
    )
    runner = ChatTurnRunner(
        build_chat_graph(models, checkpointer=InMemorySaver(serde=chat_serializer())),
        flow.factory,
        HybridRetriever(
            [ChunkRetriever(settings), CommunityRetriever(settings)],
            ConstantEmbeddings(),
        ),
        settings,
    )
    app.dependency_overrides[get_chat_runner] = lambda: runner
    try:
        document_id = await flow.upload()
        await flow.wait_status(document_id, "completed")
        thread = await flow.client.post(
            "/chat/threads", json={"document_ids": [document_id]}
        )
        assert thread.status_code == 201, thread.text
        response = await flow.client.post(
            f"/chat/threads/{thread.json()['id']}/messages",
            json={"content": "What does Acme report?"},
        )
    finally:
        app.dependency_overrides.pop(get_chat_runner, None)

    assert response.status_code == 200, response.text
    events = {}
    for block in response.text.strip().split("\n\n"):
        if block.startswith("event: "):
            event, data = block.split("\n")
            events[event.removeprefix("event: ")] = json.loads(
                data.removeprefix("data: ")
            )
    sources = events["sources"]["sources"]
    assert {source["document_id"] for source in sources} == {document_id}
    assert {source["kind"] for source in sources} == {"chunk", "community"}
    assert "Acme reports revenue" in sources[0]["excerpt"]
    assert events["done"]["message"]["citations"] == [1]
