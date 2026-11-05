"""The compiled graph with fake models, a fake retriever and InMemorySaver."""

import asyncio
from dataclasses import replace
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableGenerator
from langgraph.checkpoint.memory import InMemorySaver

from financial_assistant.agents.chat.graph import build_chat_graph
from financial_assistant.agents.chat.models import RetrievalGrade
from financial_assistant.agents.chat.nodes import AnswerUnavailableError
from financial_assistant.agents.chat.state import chat_serializer
from financial_assistant.agents.chat.tools import RetrievalUnavailableError
from financial_assistant.retrieval.types import RetrievalScope
from tests.utils import (
    FakeHybridRetriever,
    chat_context,
    chunk_source,
    community_source,
    fake_agent,
    fake_chat_models,
    tool_call_message,
)

pytestmark = pytest.mark.unit

ANSWER = [{"answer": "Revenue "}, {"answer": "Revenue grew [1].", "cited_refs": [1]}]
INSUFFICIENT = RetrievalGrade(sufficient=False, missing="The 2024 figure.")


def _graph(models):
    return build_chat_graph(models, checkpointer=InMemorySaver(serde=chat_serializer()))


def _config(thread_id):
    return {"configurable": {"thread_id": str(thread_id)}, "recursion_limit": 10}


async def _turn(graph, context, question="What was revenue?"):
    return [
        event
        async for event in graph.astream(
            {"question": question},
            config=_config(context.thread_id),
            context=context,
            stream_mode="custom",
            durability="exit",
        )
    ]


async def _state(graph, context) -> dict[str, Any]:
    return (await graph.aget_state(_config(context.thread_id))).values


async def test_one_full_turn_stores_one_question_answer_pair():
    models = fake_chat_models([tool_call_message("revenue")], answer=ANSWER)
    retriever = FakeHybridRetriever([chunk_source(1)])
    graph = _graph(models)
    context = chat_context(retriever)
    await _turn(graph, context)

    state = await _state(graph, context)
    turn_id = str(context.turn_id)
    assert [message.id for message in state["messages"]] == [f"{turn_id}:q", turn_id]
    assert state["messages"][1].content == "Revenue grew [1]."
    assert state["scratchpad"] == []
    assert state["cited_sources"][turn_id][0]["source"] == chunk_source(1)


async def test_events_arrive_in_order():
    models = fake_chat_models([tool_call_message("revenue")], answer=ANSWER)
    retriever = FakeHybridRetriever([chunk_source(1), community_source(1)])
    events = await _turn(_graph(models), chat_context(retriever))

    assert [event["type"] for event in events] == [
        "status",
        "sources",
        "status",
        "status",
        "token",
        "token",
        "answer",
    ]
    assert events[0] == {"type": "status", "stage": "searching", "query": "revenue"}
    assert [item["ref"] for item in events[1]["sources"]] == [1, 2]
    assert [event.get("stage") for event in events[2:4]] == ["grading", "answering"]
    assert events[-1]["citations"] == [1]


async def test_a_second_turn_sees_the_first_in_the_agent_prompt():
    models = fake_chat_models([tool_call_message("revenue")], answer=ANSWER)
    retriever = FakeHybridRetriever([chunk_source(1)])
    graph = _graph(models)
    first = chat_context(retriever)
    await _turn(graph, first, "First question")
    second = replace(first, turn_id=uuid4())
    await _turn(graph, second, "Follow-up question")

    request = fake_agent(models).requests[-1]
    assert [message.content for message in request[1:]] == [
        "First question",
        "Revenue grew [1].",
        "Follow-up question",
    ]
    state = await _state(graph, second)
    assert len(state["messages"]) == 4
    assert set(state["cited_sources"]) == {str(first.turn_id), str(second.turn_id)}


async def test_an_always_insufficient_grader_stops_after_the_round_cap():
    models = fake_chat_models(
        [tool_call_message("first"), tool_call_message("second")],
        grades=[INSUFFICIENT, INSUFFICIENT, INSUFFICIENT],
        answer=ANSWER,
    )
    retriever = FakeHybridRetriever([chunk_source(1)], [chunk_source(2)])
    events = await _turn(_graph(models), chat_context(retriever))

    assert [query for query, _ in retriever.calls] == ["first", "second"]
    assert len(fake_agent(models).requests) == 2
    second_round = fake_agent(models).requests[1]
    assert "The 2024 figure." in second_round[-1].content
    # Each round announces only its new sources, with refs stable across rounds.
    announced = [event for event in events if event["type"] == "sources"]
    assert [[i["ref"] for i in event["sources"]] for event in announced] == [[1], [2]]


async def test_a_round_without_tool_calls_goes_straight_to_the_answer():
    models = fake_chat_models(
        [tool_call_message("first"), AIMessage("Nothing else to search.")],
        grades=[INSUFFICIENT],
        answer=ANSWER,
    )
    retriever = FakeHybridRetriever([chunk_source(1)])
    events = await _turn(_graph(models), chat_context(retriever))
    assert len(retriever.calls) == 1
    assert events[-1]["type"] == "answer"


async def test_parallel_searches_merge_with_deterministic_refs():
    models = fake_chat_models([tool_call_message("alpha", "beta")], answer=ANSWER)
    retriever = FakeHybridRetriever(
        [chunk_source(1), chunk_source(2)], [chunk_source(2)]
    )
    graph = _graph(models)
    context = chat_context(retriever)
    events = await _turn(graph, context)
    (announced,) = [event for event in events if event["type"] == "sources"]
    assert [item["ref"] for item in announced["sources"]] == [1, 2]


async def test_filters_from_the_llm_narrow_the_scope():
    models = fake_chat_models(
        [tool_call_message("revenue", year=2025, document_type="10-K")], answer=ANSWER
    )
    retriever = FakeHybridRetriever([chunk_source(1)])
    thread_scope = RetrievalScope(user_id=1, company_ticker="NVDA")
    await _turn(_graph(models), chat_context(retriever, scope=thread_scope))
    ((_, scope),) = retriever.calls
    assert scope.user_id == 1
    assert scope.company_ticker == "NVDA"
    assert scope.year == 2025
    assert scope.document_type == "10-K"


async def test_narrowing_outside_the_thread_scope_runs_no_query():
    models = fake_chat_models(
        [tool_call_message("revenue", company_ticker="TSLA"), AIMessage("Stop.")],
        grades=[INSUFFICIENT],
        answer=[{"answer": "Not found.", "cited_refs": []}],
    )
    retriever = FakeHybridRetriever([chunk_source(1)])
    thread_scope = RetrievalScope(user_id=1, company_ticker="NVDA")
    await _turn(_graph(models), chat_context(retriever, scope=thread_scope))
    assert retriever.calls == []
    second_round = fake_agent(models).requests[1]
    tool_messages = [m for m in second_round if isinstance(m, ToolMessage)]
    assert "No documents match" in tool_messages[0].content


async def test_invalid_tool_arguments_go_back_to_the_agent_as_an_error():
    models = fake_chat_models(
        [tool_call_message("   "), tool_call_message("revenue")],
        grades=[INSUFFICIENT],
        answer=ANSWER,
    )
    retriever = FakeHybridRetriever([chunk_source(1)])
    await _turn(_graph(models), chat_context(retriever))
    assert [query for query, _ in retriever.calls] == ["revenue"]
    second_round = fake_agent(models).requests[1]
    (error,) = [m for m in second_round if isinstance(m, ToolMessage)]
    assert error.status == "error"


async def test_a_failing_search_fails_the_turn_without_reaching_the_llm():
    class BrokenRetriever(FakeHybridRetriever):
        async def search(self, session, query, scope):
            raise ConnectionError("db password in this message")

    models = fake_chat_models([tool_call_message("revenue")], answer=ANSWER)
    with pytest.raises(RetrievalUnavailableError):
        await _turn(_graph(models), chat_context(BrokenRetriever()))
    assert len(fake_agent(models).requests) == 1


async def test_a_cancelled_turn_leaves_the_history_and_the_next_turn_works():
    answering = asyncio.Event()
    hang = [True]

    async def stream(messages):
        async for _ in messages:
            pass
        if hang.pop() if hang else False:
            answering.set()
            await asyncio.Event().wait()
        for partial in ANSWER:
            yield partial

    models = replace(
        fake_chat_models([tool_call_message("revenue")]),
        answer=RunnableGenerator(stream),
    )
    retriever = FakeHybridRetriever([chunk_source(1)])
    graph = _graph(models)
    cancelled = chat_context(retriever)
    task = asyncio.create_task(_turn(graph, cancelled, "Cancelled"))
    await answering.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (await _state(graph, cancelled)).get("messages", []) == []

    next_turn = replace(cancelled, turn_id=uuid4())
    await _turn(graph, next_turn, "Next")
    state = await _state(graph, next_turn)
    assert [message.content for message in state["messages"]] == [
        "Next",
        "Revenue grew [1].",
    ]


async def test_a_failed_turn_leaves_the_history_and_the_next_turn_works():
    answers = iter(
        [
            ANSWER,
            [{"answer": "Truncated"}],  # raises AnswerUnavailableError in generate
            ANSWER,
        ]
    )

    async def stream(messages):
        async for _ in messages:
            pass
        for partial in next(answers):
            yield partial

    models = fake_chat_models([tool_call_message("revenue")])
    models = replace(models, answer=RunnableGenerator(stream))
    retriever = FakeHybridRetriever([chunk_source(1)])
    graph = _graph(models)
    first = chat_context(retriever)
    await _turn(graph, first, "First")

    failed = replace(first, turn_id=uuid4())
    with pytest.raises(AnswerUnavailableError):
        await _turn(graph, failed, "Fails")
    state = await _state(graph, failed)
    assert [message.content for message in state["messages"]] == [
        "First",
        "Revenue grew [1].",
    ]

    third = replace(first, turn_id=uuid4())
    await _turn(graph, third, "Third")
    state = await _state(graph, third)
    assert [message.content for message in state["messages"]] == [
        "First",
        "Revenue grew [1].",
        "Third",
        "Revenue grew [1].",
    ]
    assert state["sources_streamed"] == 1
