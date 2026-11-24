"""Request validation only; the endpoints themselves are integration-tested."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from financial_assistant.api.dependencies.auth import get_current_user
from financial_assistant.api.dependencies.chat import get_chat_runner
from financial_assistant.api.server import app
from financial_assistant.core.config import get_settings
from financial_assistant.core.db import get_session

pytestmark = pytest.mark.unit


# Not named `session`: tests/conftest.py marks tests using that fixture as `db`.
@pytest.fixture
def fake_session():
    return MagicMock(scalar=AsyncMock(), execute=AsyncMock(), commit=AsyncMock())


@pytest.fixture
async def client(fake_session):
    app.dependency_overrides.update(
        {
            get_current_user: lambda: SimpleNamespace(id=1),
            get_session: lambda: fake_session,
            # A lambda: FastAPI would read MagicMock's own signature as parameters.
            get_chat_runner: lambda: MagicMock(),
        }
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"content": ""},
        {"content": " \n\t "},
        {"content": "x" * (get_settings().chat_max_question_chars + 1)},
        {"content": "a\x00b"},
        {"content": "What was revenue?", "user_id": 2},
    ],
    ids=["missing", "empty", "blank", "long", "nul", "extra"],
)
async def test_invalid_messages_are_rejected_before_any_query(
    client, fake_session, body
):
    response = await client.post(f"/chat/threads/{uuid4()}/messages", json=body)
    assert response.status_code == 422
    assert [error["loc"][0] for error in response.json()["detail"]] == ["body"]
    fake_session.scalar.assert_not_awaited()


@pytest.mark.parametrize(
    "body",
    [
        {"document_ids": []},
        {"document_ids": list(range(1, 52))},
        {"document_ids": [1, 1]},
        {"document_ids": [0]},
        {"document_ids": [2**31]},
        {"title": " "},
        {"title": "x" * 201},
        {"title": "a\x00b"},
        {"company_ticker": ""},
        {"company_ticker": "A\x00"},
        {"user_id": 2},
    ],
)
async def test_invalid_threads_are_rejected_before_any_query(
    client, fake_session, body
):
    response = await client.post("/chat/threads", json=body)
    assert response.status_code == 422
    fake_session.scalar.assert_not_awaited()


@pytest.mark.parametrize(
    "query", ["limit=0", "limit=101", "offset=-1", f"offset={2**31}"]
)
async def test_invalid_pages_are_rejected(client, query):
    assert (await client.get(f"/chat/threads?{query}")).status_code == 422
