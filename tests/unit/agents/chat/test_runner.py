"""The runner with a fake graph; the turn store is replaced by recording fakes."""

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.exc import OperationalError

from financial_assistant.agents.chat.context import ScopeDocument
from financial_assistant.agents.chat.nodes import AnswerUnavailableError
from financial_assistant.agents.chat.runner import (
    NO_DOCUMENTS_ANSWER,
    ChatTurnRunner,
    cancel_running_turns,
)
from financial_assistant.agents.chat.state import merge_sources
from financial_assistant.agents.chat.tools import RetrievalUnavailableError
from financial_assistant.models.chat import ChatTurnStatus
from financial_assistant.models.document import DocumentType
from tests.utils import (
    FakeHybridRetriever,
    chat_settings,
    chunk_source,
    fake_session_factory,
    provider_error,
)

pytestmark = pytest.mark.unit

_MODULE = "financial_assistant.agents.chat.runner"
USAGE = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
SOURCES = merge_sources([], [chunk_source(1, content="Revenue was $5.")])
ANSWER = {
    "type": "answer",
    "id": "answer-id",
    "content": "Revenue was $5 [1].",
    "citations": [1],
    "sources": SOURCES,
}
DOCUMENTS = (
    ScopeDocument(
        id=1,
        filename="a.pdf",
        company_ticker="ACME",
        document_type=DocumentType.ANNUAL_REPORT,
        year=2025,
    ),
)


class FakeGraph:
    """Replays custom stream payloads; an exception in the script is raised."""

    def __init__(self, *script: Any) -> None:
        self.script = script
        self.runs: list[dict[str, Any]] = []

    async def astream(self, graph_input, **kwargs):
        self.runs.append({"input": graph_input, **kwargs})
        usage = kwargs["config"]["callbacks"][0]
        usage.usage_metadata = {"gemini-test": dict(USAGE)}
        for step in self.script:
            if isinstance(step, BaseException):
                raise step
            if step == "hang":
                await asyncio.Event().wait()
            yield step


@pytest.fixture
def finish():
    with patch(f"{_MODULE}.finish_turn", new_callable=AsyncMock) as mock:
        yield mock


@pytest.fixture(autouse=True)
def scope_documents():
    with patch(
        f"{_MODULE}.load_scope_documents", new=AsyncMock(return_value=(DOCUMENTS, 3))
    ) as mock:
        yield mock


def _runner(graph, **settings):
    return ChatTurnRunner(
        cast(Any, graph),
        fake_session_factory(),
        cast(Any, FakeHybridRetriever()),
        chat_settings(**settings),
    )


def _stream(runner, thread=None):
    thread = thread or SimpleNamespace(
        id=uuid4(), document_ids=None, company_ticker="ACME"
    )
    return runner.stream(
        user_id=7, thread=thread, turn_id=uuid4(), question="What was revenue?"
    )


async def _collect(stream):
    return [event async for event in stream]


async def test_events_and_the_done_payload(finish):
    graph = FakeGraph(
        {"type": "status", "stage": "searching", "query": "revenue"},
        {"type": "sources", "sources": SOURCES},
        {"type": "status", "stage": "grading"},
        {"type": "status", "stage": "answering"},
        {"type": "token", "text": "Revenue "},
        ANSWER,
    )
    events = await _collect(_stream(_runner(graph)))

    assert [event.event for event in events] == [
        "turn",
        "status",
        "sources",
        "status",
        "status",
        "token",
        "done",
    ]
    assert events[1].data.query == "revenue"
    source = events[2].data.sources[0]
    assert (source.ref, source.kind, source.excerpt) == (1, "chunk", "Revenue was $5.")
    done = events[-1].data
    assert done.turn_id == events[0].data.turn_id
    assert done.message.content == "Revenue was $5 [1]."
    assert done.message.citations == [1]
    assert done.message.sources == [source]
    assert (done.usage.input_tokens, done.usage.output_tokens) == (100, 20)
    finish.assert_awaited_once()
    assert finish.await_args.args[1] == done.turn_id
    assert finish.await_args.kwargs == {
        "status": ChatTurnStatus.COMPLETED,
        "usage": {"gemini-test": USAGE},
        "error_code": None,
        "title": "What was revenue?",
    }


async def test_the_graph_runs_with_the_thread_scope_and_exit_durability(finish):
    graph = FakeGraph(ANSWER)
    thread = SimpleNamespace(id=uuid4(), document_ids=[3, 4], company_ticker="ACME")
    await _collect(_stream(_runner(graph), thread))
    (run,) = graph.runs
    assert run["input"] == {"question": "What was revenue?"}
    assert run["stream_mode"] == "custom"
    assert run["durability"] == "exit"
    assert run["config"]["configurable"] == {"thread_id": str(thread.id)}
    assert run["config"]["recursion_limit"] == 10
    context = run["context"]
    assert (context.user_id, context.thread_id) == (7, thread.id)
    assert context.scope.user_id == 7
    assert context.scope.document_ids == (3, 4)
    assert context.scope.company_ticker == "ACME"
    assert context.scope_documents == DOCUMENTS
    assert context.unlisted_documents == 3


async def test_the_turn_is_recorded_before_done_reaches_the_client(finish):
    # The client may send its next message right after `done`.
    async for event in _stream(_runner(FakeGraph(ANSWER))):
        if event.event == "done":
            finish.assert_awaited_once()


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (TimeoutError(), "timeout"),
        (provider_error(429), "model_unavailable"),
        (provider_error(408), "model_unavailable"),
        (provider_error(503), "model_unavailable"),
        (provider_error(400), "model_error"),
        (RetrievalUnavailableError(), "retrieval_unavailable"),
        (OperationalError("SELECT", {}, Exception()), "retrieval_unavailable"),
        (AnswerUnavailableError(), "answer_unavailable"),
        (RuntimeError("secret detail"), "internal"),
    ],
)
async def test_failures_end_with_an_error_event_and_are_recorded(finish, error, code):
    graph = FakeGraph({"type": "status", "stage": "searching"}, error)
    events = await _collect(_stream(_runner(graph)))
    assert [event.event for event in events] == ["turn", "status", "error"]
    assert events[-1].data.code == code
    assert "secret" not in events[-1].data.message
    assert finish.await_args.kwargs["status"] == ChatTurnStatus.FAILED
    assert finish.await_args.kwargs["error_code"] == code
    assert finish.await_args.kwargs["usage"] == {"gemini-test": USAGE}


async def test_the_turn_deadline_fails_a_hanging_run(finish):
    runner = _runner(
        FakeGraph("hang"), chat_turn_timeout_seconds=0.01, chat_turn_stale_seconds=1
    )
    events = await _collect(_stream(runner))
    assert events[-1].data.code == "timeout"
    assert finish.await_args.kwargs["error_code"] == "timeout"


async def test_a_run_without_an_answer_is_a_failure(finish):
    events = await _collect(_stream(_runner(FakeGraph())))
    assert events[-1].data.code == "answer_unavailable"


async def test_closing_the_stream_cancels_the_run_and_records_partial_usage(finish):
    graph = FakeGraph({"type": "token", "text": "Reve"}, "hang")
    stream = _stream(_runner(graph))
    assert (await anext(stream)).event == "turn"
    assert (await anext(stream)).event == "token"
    await stream.aclose()
    await cancel_running_turns()
    finish.assert_awaited_once()
    assert finish.await_args.kwargs["status"] == ChatTurnStatus.CANCELLED
    assert finish.await_args.kwargs["usage"] == {"gemini-test": USAGE}
    assert finish.await_args.kwargs["error_code"] is None


async def test_shutdown_cancels_running_turns_and_lets_them_finish(finish):
    started = asyncio.Event()

    class SignallingGraph(FakeGraph):
        async def astream(self, graph_input, **kwargs):
            started.set()
            await asyncio.Event().wait()
            yield {}

    stream = _stream(_runner(SignallingGraph()))
    consumer = asyncio.create_task(_collect(stream))
    await started.wait()
    await cancel_running_turns()
    assert finish.await_args.kwargs["status"] == ChatTurnStatus.CANCELLED
    await consumer


async def test_no_documents_in_scope_answers_without_running_the_graph(
    finish, scope_documents
):
    scope_documents.return_value = ((), 0)
    graph = FakeGraph()
    events = await _collect(_stream(_runner(graph)))
    assert graph.runs == []
    assert [event.event for event in events] == ["turn", "token", "done"]
    done = events[-1].data
    assert done.message.content == NO_DOCUMENTS_ANSWER
    assert done.message.sources == []
    assert (done.usage.input_tokens, done.usage.output_tokens) == (0, 0)
    assert finish.await_args.kwargs["status"] == ChatTurnStatus.COMPLETED
    assert finish.await_args.kwargs["usage"] == {}


async def test_a_failure_to_record_the_turn_is_reported_to_the_client(finish):
    finish.side_effect = ConnectionError("database is down")
    events = await _collect(_stream(_runner(FakeGraph(ANSWER))))
    assert [event.event for event in events] == ["turn", "error"]
    assert events[-1].data.code == "internal"
