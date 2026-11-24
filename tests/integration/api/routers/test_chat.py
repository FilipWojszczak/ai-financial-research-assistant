"""The chat API with the real runner and graph, fake models and InMemorySaver."""

import asyncio
import json
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from anyio import BrokenResourceError
from httpx import AsyncClient
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from financial_assistant.agents.chat.graph import build_chat_graph
from financial_assistant.agents.chat.models import ChatModels
from financial_assistant.agents.chat.runner import (
    NO_DOCUMENTS_ANSWER,
    ChatTurnRunner,
    cancel_running_turns,
)
from financial_assistant.api.dependencies.chat import get_chat_runner
from financial_assistant.api.server import app
from financial_assistant.core.config import get_settings
from financial_assistant.models import ChatThread, ChatTurn, User
from financial_assistant.models.chat import ChatTurnStatus
from financial_assistant.models.document import DocumentStatus
from financial_assistant.schemas.chat import ChatEvent, TokenEvent
from tests.utils import (
    DocumentFactory,
    FakeHybridRetriever,
    TokenFactory,
    UserFactory,
    chunk_source,
    fake_agent,
    fake_chat_models,
    tool_call_message,
)

pytestmark = pytest.mark.integration

ANSWER = [{"answer": "Revenue "}, {"answer": "Revenue grew [1].", "cited_refs": [1]}]
QUESTION = {"content": "What was revenue?"}


def _models() -> ChatModels:
    return fake_chat_models([tool_call_message("revenue")], answer=ANSWER)


@pytest.fixture
def use_runner(
    client: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
    chat_checkpointer: InMemorySaver,
) -> Callable[..., ChatTurnRunner]:
    """Serve the API with a real runner over the given fakes."""

    def _use(models: ChatModels | None = None, retriever: Any = None):
        graph = build_chat_graph(models or _models(), checkpointer=chat_checkpointer)
        runner = ChatTurnRunner(
            graph,
            session_factory,
            cast(Any, retriever or FakeHybridRetriever([chunk_source(1)])),
            get_settings(),
        )
        app.dependency_overrides[get_chat_runner] = lambda: runner
        return runner

    return _use


@pytest.fixture
async def user(user_factory: UserFactory) -> User:
    return await user_factory(email="chatter@example.com")


@pytest.fixture
def headers(user: User, token_factory: TokenFactory) -> dict[str, str]:
    return {"Authorization": f"Bearer {token_factory(user)}"}


@pytest.fixture
async def thread(session: AsyncSession, user: User) -> ChatThread:
    return await _add_thread(session, user)


async def _add_thread(session: AsyncSession, user: User, **fields) -> ChatThread:
    thread = ChatThread(user_id=user.id, **fields)
    session.add(thread)
    await session.commit()
    return thread


async def _add_turn(session: AsyncSession, thread: ChatThread, **fields) -> ChatTurn:
    turn = ChatTurn(thread_id=thread.id, user_id=thread.user_id, **fields)
    session.add(turn)
    await session.commit()
    return turn


def _messages_url(thread: ChatThread) -> str:
    return f"/chat/threads/{thread.id}/messages"


def _parse_sse(text: str) -> list[tuple[str, Any]]:
    """The events as a client parses them: data lines are joined with newlines."""
    events = []
    for block in text.split("\n\n"):
        fields: dict[str, list[str]] = {"event": [], "data": []}
        for line in block.split("\n"):
            name, _, value = line.partition(": ")
            if name in fields:
                fields[name].append(value)
        if fields["event"]:
            events.append((fields["event"][0], json.loads("\n".join(fields["data"]))))
    return events


async def _turn_count(session: AsyncSession, user: User) -> int | None:
    return await session.scalar(
        select(func.count()).select_from(ChatTurn).where(ChatTurn.user_id == user.id)
    )


async def _turn_status(session: AsyncSession, user: User):
    result = await session.execute(
        select(ChatTurn.status, ChatTurn.error_code).where(ChatTurn.user_id == user.id)
    )
    return tuple(result.one())


# POST /chat/threads


async def test_create_thread(
    client: AsyncClient, user: User, headers, document_factory: DocumentFactory
):
    own = await document_factory(owner_id=user.id, status=DocumentStatus.PROCESSING)
    public = await document_factory(owner_id=None)
    body = {
        "title": " Apple ",
        "document_ids": [own.id, public.id],
        "company_ticker": "AAPL",
    }
    response = await client.post("/chat/threads", json=body, headers=headers)
    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "Apple"
    assert data["document_ids"] == [own.id, public.id]
    assert data["company_ticker"] == "AAPL"
    assert UUID(data["id"])


async def test_create_thread_without_filters(client: AsyncClient, headers):
    response = await client.post("/chat/threads", json={}, headers=headers)
    assert response.status_code == 201
    data = response.json()
    assert (data["title"], data["document_ids"]) == (None, None)


async def test_create_thread_with_an_inaccessible_document(
    client: AsyncClient,
    user: User,
    headers,
    user_factory: UserFactory,
    document_factory: DocumentFactory,
):
    other = await user_factory(email="other@example.com")
    own = await document_factory(owner_id=user.id)
    private = await document_factory(owner_id=other.id)
    for hidden in (private.id, 2**31 - 1):
        response = await client.post(
            "/chat/threads", json={"document_ids": [own.id, hidden]}, headers=headers
        )
        assert response.status_code == 404
        assert response.json() == {"detail": "Document not found"}


# GET /chat/threads


async def test_list_returns_own_threads_newest_first(
    client: AsyncClient,
    session: AsyncSession,
    user: User,
    headers,
    user_factory: UserFactory,
):
    now = datetime.now(UTC)
    old = await _add_thread(session, user, updated_at=now - timedelta(days=2))
    new = await _add_thread(session, user, updated_at=now)
    middle = await _add_thread(session, user, updated_at=now - timedelta(days=1))
    await _add_thread(session, await user_factory(email="other@example.com"))

    response = await client.get("/chat/threads", headers=headers)
    expected = [str(item.id) for item in (new, middle, old)]
    assert [item["id"] for item in response.json()] == expected

    page = await client.get("/chat/threads?limit=1&offset=1", headers=headers)
    assert [item["id"] for item in page.json()] == [str(middle.id)]


# GET /chat/threads/{id}


async def test_get_thread_returns_the_history_with_sources(
    client: AsyncClient,
    session: AsyncSession,
    user: User,
    thread: ChatThread,
    headers,
    use_runner,
    document_factory: DocumentFactory,
):
    await document_factory()
    use_runner()
    await client.post(_messages_url(thread), json=QUESTION, headers=headers)

    response = await client.get(f"/chat/threads/{thread.id}", headers=headers)
    assert response.status_code == 200
    turn_id = await session.scalar(
        select(ChatTurn.id).where(ChatTurn.user_id == user.id)
    )
    question, answer = response.json()["messages"]
    assert question["id"] == f"{turn_id}:q"
    assert (question["role"], question["content"]) == ("user", "What was revenue?")
    assert question["sources"] is None
    assert answer["id"] == str(turn_id)
    assert (answer["role"], answer["content"]) == ("assistant", "Revenue grew [1].")
    assert [(item["ref"], item["excerpt"]) for item in answer["sources"]] == [
        (1, "Text")
    ]
    assert question["created_at"] and answer["created_at"]
    title = await session.scalar(
        select(ChatThread.title).where(ChatThread.id == thread.id)
    )
    assert title == "What was revenue?"


async def test_a_new_thread_has_no_messages(
    client: AsyncClient, thread: ChatThread, headers, use_runner
):
    use_runner()
    response = await client.get(f"/chat/threads/{thread.id}", headers=headers)
    assert response.json()["messages"] == []


async def test_another_users_thread_is_not_found(
    client: AsyncClient,
    session: AsyncSession,
    user: User,
    headers,
    use_runner,
    user_factory: UserFactory,
):
    use_runner()
    foreign = await _add_thread(session, await user_factory(email="other@example.com"))
    for thread_id in (foreign.id, uuid4()):
        url = f"/chat/threads/{thread_id}"
        for response in (
            await client.get(url, headers=headers),
            await client.delete(url, headers=headers),
            await client.post(f"{url}/messages", json=QUESTION, headers=headers),
        ):
            assert response.status_code == 404
            assert response.json() == {"detail": "Thread not found"}
    assert await _turn_count(session, user) == 0
    assert await session.scalar(
        select(ChatThread.id).where(ChatThread.id == foreign.id)
    )


# DELETE /chat/threads/{id}


async def test_delete_removes_checkpoints_and_keeps_turns_for_the_budget(
    client: AsyncClient,
    session: AsyncSession,
    user: User,
    thread: ChatThread,
    headers,
    use_runner,
    chat_checkpointer: InMemorySaver,
    document_factory: DocumentFactory,
):
    await document_factory()
    use_runner()
    await client.post(_messages_url(thread), json=QUESTION, headers=headers)
    config = {"configurable": {"thread_id": str(thread.id)}}
    assert await chat_checkpointer.aget_tuple(cast(Any, config)) is not None

    response = await client.delete(f"/chat/threads/{thread.id}", headers=headers)

    assert response.status_code == 204
    assert await chat_checkpointer.aget_tuple(cast(Any, config)) is None
    assert (
        await session.scalar(select(ChatThread.id).where(ChatThread.id == thread.id))
        is None
    )
    turns = await session.execute(
        select(ChatTurn.thread_id).where(ChatTurn.user_id == user.id)
    )
    assert turns.scalars().all() == [None]


async def test_delete_while_a_turn_is_running(
    client: AsyncClient, session: AsyncSession, thread: ChatThread, headers
):
    await _add_turn(session, thread)
    response = await client.delete(f"/chat/threads/{thread.id}", headers=headers)
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "thread_busy"
    assert int(response.headers["Retry-After"]) > 0


async def test_delete_ignores_an_abandoned_turn(
    client: AsyncClient, session: AsyncSession, thread: ChatThread, headers
):
    await _add_turn(session, thread, started_at=datetime.now(UTC) - timedelta(hours=1))
    response = await client.delete(f"/chat/threads/{thread.id}", headers=headers)
    assert response.status_code == 204


# POST /chat/threads/{id}/messages


async def test_message_streams_events_in_order(
    client: AsyncClient,
    session: AsyncSession,
    user: User,
    thread: ChatThread,
    headers,
    use_runner,
    document_factory: DocumentFactory,
):
    await document_factory()
    use_runner()
    response = await client.post(_messages_url(thread), json=QUESTION, headers=headers)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(response.text)
    assert [name for name, _ in events] == [
        "turn",
        "status",
        "sources",
        "status",
        "status",
        "token",
        "token",
        "done",
    ]
    assert [data["stage"] for name, data in events if name == "status"] == [
        "searching",
        "grading",
        "answering",
    ]
    assert events[2][1]["sources"][0]["ref"] == 1
    done = events[-1][1]
    assert done["turn_id"] == events[0][1]["turn_id"] == done["message"]["id"]
    assert done["message"]["content"] == "Revenue grew [1]."
    assert done["message"]["citations"] == [1]
    assert await _turn_status(session, user) == (ChatTurnStatus.COMPLETED, None)


async def test_a_follow_up_sees_the_previous_turn(
    client: AsyncClient,
    thread: ChatThread,
    headers,
    use_runner,
    document_factory: DocumentFactory,
):
    await document_factory()
    models = _models()
    use_runner(models)
    for content in ("First", "Second"):
        await client.post(
            _messages_url(thread), json={"content": content}, headers=headers
        )
    prompt = [message.content for message in fake_agent(models).requests[-1]]
    assert prompt[1:4] == ["First", "Revenue grew [1].", "Second"]


async def test_line_separators_do_not_split_an_event(
    client: AsyncClient,
    thread: ChatThread,
    headers,
    use_runner,
    document_factory: DocumentFactory,
):
    # str.splitlines(), which FastAPI uses to write data lines, splits on these.
    content = "a\N{LINE SEPARATOR}b\N{PARAGRAPH SEPARATOR}c\u0085d"
    await document_factory()
    use_runner(retriever=FakeHybridRetriever([chunk_source(1, content=content)]))
    response = await client.post(_messages_url(thread), json=QUESTION, headers=headers)
    events = dict(_parse_sse(response.text))
    assert events["sources"]["sources"][0]["excerpt"] == content


async def test_nothing_in_scope_gets_the_fixed_answer(
    client: AsyncClient, session: AsyncSession, user: User, headers, use_runner
):
    thread = await _add_thread(session, user, company_ticker="NO-SUCH-TICKER")
    models = _models()
    use_runner(models)
    response = await client.post(_messages_url(thread), json=QUESTION, headers=headers)
    events = dict(_parse_sse(response.text))
    assert events["done"]["message"]["content"] == NO_DOCUMENTS_ANSWER
    assert events["done"]["usage"] == {"input_tokens": 0, "output_tokens": 0}
    assert fake_agent(models).requests == []


async def test_a_failure_after_the_first_byte_is_an_error_event(
    client: AsyncClient,
    session: AsyncSession,
    user: User,
    thread: ChatThread,
    headers,
    use_runner,
    document_factory: DocumentFactory,
):
    class BrokenRetriever:
        async def search(self, session, query, scope):
            raise ConnectionError("secret detail")

    await document_factory()
    use_runner(retriever=BrokenRetriever())
    response = await client.post(_messages_url(thread), json=QUESTION, headers=headers)
    assert response.status_code == 200
    name, data = _parse_sse(response.text)[-1]
    assert (name, data["code"]) == ("error", "retrieval_unavailable")
    assert "secret" not in response.text
    assert await _turn_status(session, user) == (
        ChatTurnStatus.FAILED,
        "retrieval_unavailable",
    )


async def test_a_busy_thread_is_refused(
    client: AsyncClient,
    session: AsyncSession,
    user: User,
    thread: ChatThread,
    headers,
    use_runner,
):
    use_runner()
    await _add_turn(session, thread)
    response = await client.post(_messages_url(thread), json=QUESTION, headers=headers)
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "thread_busy"
    assert 0 < int(response.headers["Retry-After"]) <= 361
    assert await _turn_count(session, user) == 1


async def test_a_full_thread_is_refused(
    client: AsyncClient,
    session: AsyncSession,
    user: User,
    thread: ChatThread,
    headers,
    use_runner,
):
    use_runner()
    limit = get_settings().chat_max_turns_per_thread
    session.add_all(
        ChatTurn(thread_id=thread.id, user_id=user.id, status=ChatTurnStatus.COMPLETED)
        for _ in range(limit)
    )
    await session.commit()
    response = await client.post(_messages_url(thread), json=QUESTION, headers=headers)
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "thread_full"
    assert await _turn_count(session, user) == limit


async def test_an_exhausted_budget_is_refused(
    client: AsyncClient,
    session: AsyncSession,
    user: User,
    thread: ChatThread,
    headers,
    use_runner,
):
    use_runner()
    await _add_turn(
        session,
        thread,
        status=ChatTurnStatus.COMPLETED,
        input_tokens=get_settings().chat_daily_token_budget,
        started_at=datetime.now(UTC),
    )
    response = await client.post(_messages_url(thread), json=QUESTION, headers=headers)
    assert response.status_code == 429
    assert response.json()["detail"]["code"] == "budget_exceeded"
    assert 0 < int(response.headers["Retry-After"]) <= 24 * 60 * 60
    assert await _turn_count(session, user) == 1


async def test_blank_content_is_rejected_and_claims_no_turn(
    client: AsyncClient,
    session: AsyncSession,
    user: User,
    thread: ChatThread,
    headers,
    use_runner,
):
    use_runner()
    response = await client.post(
        _messages_url(thread), json={"content": "  "}, headers=headers
    )
    assert response.status_code == 422
    assert await _turn_count(session, user) == 0


async def test_every_endpoint_requires_authentication(
    client: AsyncClient, thread: ChatThread, use_runner
):
    use_runner()
    url = f"/chat/threads/{thread.id}"
    for response in (
        await client.post("/chat/threads", json={}),
        await client.get("/chat/threads"),
        await client.get(url),
        await client.delete(url),
        await client.post(f"{url}/messages", json=QUESTION),
    ):
        assert response.status_code == 401


# Client disconnects: httpx buffers the whole response, so call the app directly.


async def _post_then_disconnect(
    thread: ChatThread, headers: dict[str, str], *, block_send: bool
) -> None:
    """Send the message and disconnect after the first body chunk."""
    body = json.dumps(QUESTION).encode()
    requests = iter([{"type": "http.request", "body": body, "more_body": False}])
    first_chunk = asyncio.Event()

    async def receive():
        if (message := next(requests, None)) is not None:
            return message
        await first_chunk.wait()
        # Let the events that are ready pile up behind a blocked send.
        await asyncio.sleep(0.1)
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            first_chunk.set()
            if block_send:
                await asyncio.Event().wait()

    scope = {
        "type": "http",
        # What uvicorn declares: Starlette then listens for http.disconnect.
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": _messages_url(thread),
        "raw_path": _messages_url(thread).encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"content-type", b"application/json"),
            (b"authorization", headers["Authorization"].encode()),
        ],
        "client": ("127.0.0.1", 1),
        "server": ("test", 80),
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=10)


async def test_a_client_disconnect_cancels_the_turn(
    client: AsyncClient,
    session: AsyncSession,
    user: User,
    thread: ChatThread,
    headers,
    use_runner,
    chat_checkpointer: InMemorySaver,
    document_factory: DocumentFactory,
):
    class HangingRetriever:
        async def search(self, session, query, scope):
            await asyncio.Event().wait()

    await document_factory()
    use_runner(retriever=HangingRetriever())
    await _post_then_disconnect(thread, headers, block_send=False)
    await cancel_running_turns()  # Already cancelled; waits for its bookkeeping.

    assert await _turn_status(session, user) == (ChatTurnStatus.CANCELLED, None)
    history = await client.get(f"/chat/threads/{thread.id}", headers=headers)
    assert history.json()["messages"] == []


@pytest.mark.parametrize("block_send", [False, True])
async def test_a_client_disconnect_closes_the_event_stream(
    client: AsyncClient, thread: ChatThread, headers, block_send
):
    closed = False

    class EndlessRunner:
        async def stream(self, **kwargs):
            nonlocal closed
            try:
                while True:
                    yield ChatEvent("token", TokenEvent(text="x"))
                    await asyncio.sleep(0)
            finally:
                closed = True

    app.dependency_overrides[get_chat_runner] = EndlessRunner
    # FastAPI 0.135.2 itself may raise this (in an ExceptionGroup): it closes its
    # keep-alive stream before cancelling the task that writes to it.
    with suppress(BrokenResourceError):
        await _post_then_disconnect(thread, headers, block_send=block_send)
    # By the time the request ends, not at some later garbage collection.
    assert closed


# DELETE /auth/me


@pytest.mark.parametrize("owns_document", [False, True], ids=["hard", "tombstone"])
async def test_deleting_the_account_deletes_its_chats(
    client: AsyncClient,
    session: AsyncSession,
    user: User,
    thread: ChatThread,
    headers,
    use_runner,
    chat_checkpointer: InMemorySaver,
    document_factory: DocumentFactory,
    owns_document: bool,
):
    await document_factory(owner_id=user.id if owns_document else None)
    use_runner()
    await client.post(_messages_url(thread), json=QUESTION, headers=headers)
    user_id, thread_id = user.id, thread.id

    response = await client.request(
        "DELETE", "/auth/me", json={"password": "securepassword"}, headers=headers
    )

    assert response.status_code == 204
    config = {"configurable": {"thread_id": str(thread_id)}}
    assert await chat_checkpointer.aget_tuple(cast(Any, config)) is None
    for model in (ChatThread, ChatTurn):
        assert (
            await session.scalar(
                select(func.count()).select_from(model).where(model.user_id == user_id)
            )
            == 0
        )
    remaining = await session.scalar(select(User.id).where(User.id == user_id))
    assert (remaining is not None) == owns_document


# The endpoint's rollback also ends the transaction the `session` fixture rolls back.
@pytest.mark.filterwarnings("ignore:transaction already deassociated")
async def test_deleting_the_account_while_a_turn_is_running(
    client: AsyncClient, session: AsyncSession, thread: ChatThread, headers
):
    await _add_turn(session, thread)
    response = await client.request(
        "DELETE", "/auth/me", json={"password": "securepassword"}, headers=headers
    )
    assert response.status_code == 409
    assert "still being answered" in response.json()["detail"]
