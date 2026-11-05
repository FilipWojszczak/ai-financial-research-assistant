"""Nodes as plain functions: state in, partial update out."""

from dataclasses import replace
from typing import Any

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableGenerator
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from langgraph.types import Overwrite

from financial_assistant.agents.chat.models import RetrievalGrade
from financial_assistant.agents.chat.nodes import (
    AnswerUnavailableError,
    ChatNodes,
    context_sources,
    prepare,
    route_after_agent,
    route_after_grade,
)
from financial_assistant.agents.chat.state import merge_sources
from financial_assistant.agents.chat.tools import CHAT_TOOLS
from tests.utils import (
    chat_context,
    chunk_source,
    community_source,
    fake_agent,
    fake_chat_models,
    tool_call_message,
)

pytestmark = pytest.mark.unit


def _state(**overrides: Any) -> Any:
    return {
        "messages": [],
        "scratchpad": [],
        "question": "What was revenue?",
        "sources": [],
        "sources_streamed": 0,
        "round": 0,
        "grade": None,
        "cited_sources": {},
        **overrides,
    }


def _runtime(**settings: Any):
    written: list[dict[str, Any]] = []
    context = chat_context(**settings)
    return Runtime(context=context, stream_writer=written.append), written


def _types(written) -> list[str]:
    return [event["type"] for event in written]


def test_prepare_resets_the_per_turn_fields():
    update = prepare(_state())
    assert update["scratchpad"] == [RemoveMessage(id=REMOVE_ALL_MESSAGES)]
    assert isinstance(update["sources"], Overwrite)
    assert update["sources"].value == []
    assert update["sources_streamed"] == 0
    assert update["round"] == 0
    assert update["grade"] is None


@pytest.mark.parametrize(("round_", "choice"), [(0, "any"), (1, "auto")])
async def test_agent_forces_a_search_only_in_the_first_round(round_, choice):
    models = fake_chat_models([tool_call_message("revenue")])
    runtime, _ = _runtime()
    update = await ChatNodes(models, CHAT_TOOLS).agent(_state(round=round_), runtime)
    assert fake_agent(models).bound_tool_choices == [choice]
    assert update["round"] == round_ + 1
    assert len(update["scratchpad"]) == 1


async def test_agent_prompt_has_documents_history_question_and_scratchpad():
    models = fake_chat_models([tool_call_message("revenue")])
    runtime, _ = _runtime(chat_history_turns=1)
    history = [
        HumanMessage("old question"),
        AIMessage("old answer"),
        HumanMessage("previous question"),
        AIMessage("previous answer"),
    ]
    feedback = HumanMessage("Reviewer: search again")
    await ChatNodes(models, CHAT_TOOLS).agent(
        _state(messages=history, scratchpad=[feedback]), runtime
    )
    request = fake_agent(models).requests[0]
    assert "- ACME 10-K 2025 (a.pdf)" in request[0].content
    assert [message.content for message in request[1:]] == [
        "previous question",
        "previous answer",
        "What was revenue?",
        "Reviewer: search again",
    ]


async def test_agent_drops_tool_calls_beyond_the_limit():
    models = fake_chat_models([tool_call_message("a1", "b2", "c3", "d4")])
    runtime, _ = _runtime(chat_max_tool_calls_per_round=2)
    update = await ChatNodes(models, CHAT_TOOLS).agent(_state(), runtime)
    calls = update["scratchpad"][0].tool_calls
    assert [call["args"]["query"] for call in calls] == ["a1", "b2"]


def test_route_after_agent():
    with_calls = _state(scratchpad=[tool_call_message("revenue")])
    assert route_after_agent(with_calls) == "tools"
    assert route_after_agent(_state(scratchpad=[AIMessage("done")])) == "generate"


@pytest.mark.parametrize(
    ("sufficient", "round_", "route"),
    [(True, 1, "generate"), (False, 1, "agent"), (False, 2, "generate")],
)
def test_route_after_grade(sufficient, round_, route):
    runtime, _ = _runtime(chat_max_retrieval_rounds=2)
    state = _state(grade=RetrievalGrade(sufficient=sufficient), round=round_)
    assert route_after_grade(state, runtime) == route


def test_context_sources_cap_chunks_but_keep_communities():
    sources = merge_sources(
        [], [chunk_source(1), community_source(1), chunk_source(2), chunk_source(3)]
    )
    assert [item["ref"] for item in context_sources(sources, 2)] == [1, 2, 3]


async def test_grade_streams_only_new_sources_and_records_them():
    sources = merge_sources([], [chunk_source(1), chunk_source(2)])
    models = fake_chat_models([], grades=[RetrievalGrade(sufficient=True)])
    runtime, written = _runtime()
    update = await ChatNodes(models, CHAT_TOOLS).grade(
        _state(sources=sources, sources_streamed=1), runtime
    )
    assert _types(written) == ["sources", "status"]
    assert [item["ref"] for item in written[0]["sources"]] == [2]
    assert written[1] == {"type": "status", "stage": "grading"}
    assert update["sources_streamed"] == 2
    assert update["grade"].sufficient is True
    assert "scratchpad" not in update


async def test_insufficient_grade_tells_the_agent_what_is_missing():
    grade = RetrievalGrade(sufficient=False, missing="The 2024 figure.")
    models = fake_chat_models([], grades=[grade])
    runtime, written = _runtime()
    update = await ChatNodes(models, CHAT_TOOLS).grade(_state(), runtime)
    assert update["grade"] == grade
    assert "The 2024 figure." in update["scratchpad"][0].content
    assert _types(written) == ["status"]  # no new sources


async def test_unparseable_grade_counts_as_sufficient(caplog):
    models = fake_chat_models([], grades=[OutputParserException("empty")])
    runtime, _ = _runtime()
    update = await ChatNodes(models, CHAT_TOOLS).grade(_state(), runtime)
    assert update["grade"].sufficient is True
    assert "not parseable" in caplog.text


async def test_generate_streams_deltas_and_appends_one_question_answer_pair():
    sources = merge_sources([], [chunk_source(1, content="x" * 500), chunk_source(2)])
    models = fake_chat_models(
        [],
        answer=[
            {"answer": "Reve"},
            {"answer": "Revenue was [1]"},
            {"answer": "Revenue was [1].", "cited_refs": [1, 7]},
        ],
    )
    runtime, written = _runtime()
    turn_id = str(runtime.context.turn_id)
    update = await ChatNodes(models, CHAT_TOOLS).generate(
        _state(sources=sources), runtime
    )

    assert written[0] == {"type": "status", "stage": "answering"}
    assert [e["text"] for e in written if e["type"] == "token"] == [
        "Reve",
        "nue was [1]",
        ".",
    ]
    answer = written[-1]
    assert answer["type"] == "answer"
    assert answer["id"] == turn_id
    assert answer["content"] == "Revenue was [1]."
    assert answer["citations"] == [1]  # the unknown ref 7 is dropped
    assert [item["ref"] for item in answer["sources"]] == [1]

    question, reply = update["messages"]
    assert (question.id, question.content) == (f"{turn_id}:q", "What was revenue?")
    assert (reply.id, reply.content) == (turn_id, "Revenue was [1].")
    assert update["scratchpad"] == [RemoveMessage(id=REMOVE_ALL_MESSAGES)]
    # History keeps a snapshot of the excerpt, not the whole parent chunk.
    (snapshot,) = update["cited_sources"][turn_id]
    assert snapshot["ref"] == 1
    assert len(snapshot["source"].content) == 300


async def test_generate_skips_a_partial_that_is_not_a_prefix_extension():
    models = fake_chat_models(
        [],
        answer=[
            {"answer": "Revenue"},
            {"answer": "Rewritten"},
            {"answer": "Revenue grew", "cited_refs": []},
        ],
    )
    runtime, written = _runtime()
    update = await ChatNodes(models, CHAT_TOOLS).generate(_state(), runtime)
    tokens = [event["text"] for event in written if event["type"] == "token"]
    assert tokens == ["Revenue", " grew"]
    assert update["messages"][1].content == "Revenue grew"


async def test_generate_tells_the_model_what_the_grader_found_missing():
    seen = []

    async def record(messages):
        seen.extend([item async for item in messages])
        yield {"answer": "Not found.", "cited_refs": []}

    models = replace(fake_chat_models([]), answer=RunnableGenerator(record))
    runtime, _ = _runtime()
    grade = RetrievalGrade(sufficient=False, missing="The 2024 figure.")
    await ChatNodes(models, CHAT_TOOLS).generate(_state(grade=grade), runtime)
    assert "missing from the sources: The 2024 figure." in seen[0][-1].content


@pytest.mark.parametrize(
    "partials",
    [
        [],  # e.g. a safety block
        [{"answer": "   ", "cited_refs": []}],
        [{"answer": "Truncated JSON without citations"}],
    ],
)
async def test_generate_raises_when_there_is_no_usable_answer(partials):
    models = fake_chat_models([], answer=partials)
    runtime, written = _runtime()
    with pytest.raises(AnswerUnavailableError):
        await ChatNodes(models, CHAT_TOOLS).generate(_state(), runtime)
    assert "answer" not in _types(written)
