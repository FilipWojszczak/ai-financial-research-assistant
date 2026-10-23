"""
Smoke test of the real chat models (paid, ~$0.01): model IDs, forced tool calls,
a round-2 agent call with tool results, the grader's structured output, partial
JSON streaming of the answer, token usage including thinking, thinking levels.
Run with: RUN_LLM_TESTS=1 uv run pytest tests/llm/agents/chat/test_models_smoke.py -s
"""

from itertools import pairwise

import pytest
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from financial_assistant.agents.chat.models import (
    LOWEST_THINKING_LEVEL,
    ChatModels,
    GeneratedAnswer,
    RetrievalGrade,
    create_chat_models,
)
from financial_assistant.core.config import get_settings

pytestmark = pytest.mark.llm

QUESTION = "What was Northwind's total revenue in fiscal 2025, and how did it change?"
SOURCES = (
    "<sources>\n"
    '<source ref="1" kind="chunk" document="NWND 10-K 2025 (northwind_10k.pdf)">'
    "Northwind Traders reported total revenue of $412.7 million for fiscal 2025, up "
    "8.0% from $382.1 million in fiscal 2024. The Logistics segment contributed "
    "$251.3 million and the Retail segment $161.4 million.</source>\n"
    "</sources>"
)


class SearchArgs(BaseModel):
    query: str = Field(description="What to search for in the documents")


@tool("search_documents", args_schema=SearchArgs)
def search_documents(query: str) -> str:
    """Search the user's financial documents."""
    return ""


@pytest.fixture
def models() -> ChatModels:
    # Function-scoped: the async clients must not outlive one test's event loop.
    return create_chat_models(get_settings())


def _usage_of(usage: UsageMetadataCallbackHandler, model: str):
    print(f"\n{model}: {usage.usage_metadata}")
    matching = [value for name, value in usage.usage_metadata.items() if model in name]
    assert matching, f"no usage reported for {model}: {usage.usage_metadata}"
    total = matching[0]
    assert total["input_tokens"] > 0
    assert total["output_tokens"] > 0
    return total


async def test_agent_is_forced_to_search_then_reads_the_results(models):
    usage = UsageMetadataCallbackHandler()
    config = {"callbacks": [usage]}
    messages = [
        SystemMessage(
            "You are a financial research assistant. Call search_documents for any "
            "factual question."
        ),
        HumanMessage(QUESTION),
    ]
    first = await models.agent.bind_tools(
        [search_documents], tool_choice="any"
    ).ainvoke(messages, config=config)
    assert isinstance(first, AIMessage)
    assert first.tool_calls
    assert {call["name"] for call in first.tool_calls} == {"search_documents"}

    # Round 2 sends the tool calls back with their results and a reviewer note.
    # Gemini 3 rejects function calls returned without their thought signatures.
    round_two = [
        *messages,
        first,
        *(
            ToolMessage(
                "[1] NWND 10-K 2025: total revenue $412.7 million",
                tool_call_id=call["id"],
            )
            for call in first.tool_calls
        ),
        HumanMessage(
            "Reviewer: the year-over-year change is missing; search again with a "
            "different query."
        ),
    ]
    second = await models.agent.bind_tools(
        [search_documents], tool_choice="auto"
    ).ainvoke(round_two, config=config)
    assert isinstance(second, AIMessage)
    print(f"\nround 2 tool calls: {second.tool_calls}")
    _usage_of(usage, get_settings().chat_agent_model)


async def test_grader_returns_a_retrieval_grade(models):
    usage = UsageMetadataCallbackHandler()
    grade = await models.grader.ainvoke(
        [
            SystemMessage(
                "Decide whether the numbered sources contain what the question needs."
            ),
            HumanMessage(f"Question: {QUESTION}\n\n{SOURCES}"),
        ],
        config={"callbacks": [usage]},
    )
    print(f"\ngrade: {grade!r}")
    assert isinstance(grade, RetrievalGrade)
    _usage_of(usage, get_settings().chat_grader_model)


async def test_answer_streams_growing_partial_dicts(models):
    usage = UsageMetadataCallbackHandler()
    partials = [
        partial
        async for partial in models.answer.astream(
            [
                SystemMessage(
                    "Answer only from the sources in 3-4 sentences. Cite sources "
                    "inline as [n] and list the refs you cite."
                ),
                HumanMessage(f"{SOURCES}\n\nQuestion: {QUESTION}"),
            ],
            config={"callbacks": [usage]},
        )
    ]
    answers = [partial.get("answer", "") for partial in partials]
    non_prefix = sum(
        1 for earlier, later in pairwise(answers) if not later.startswith(earlier)
    )
    print(
        f"\n{len(partials)} partials, {len(set(answers))} distinct answer values, "
        f"{non_prefix} not prefix extensions; first: {answers[:3]!r}"
    )
    # The generate node streams deltas of `answer`; that needs several partials.
    assert len({answer for answer in answers if answer}) > 1
    final = GeneratedAnswer.model_validate(partials[-1])
    assert "412.7" in final.answer
    assert final.cited_refs == [1]
    settings = get_settings()
    total = _usage_of(usage, settings.chat_answer_model)
    # Thinking tokens are billed as output (output_tokens includes them), so they
    # count against the budget; the thinking level must leave room under the cap.
    reasoning = total.get("output_token_details", {}).get("reasoning", 0)
    print(f"thinking_level={settings.chat_answer_thinking_level}: {reasoning=}")
    assert reasoning <= total["output_tokens"]
    assert total["output_tokens"] <= settings.chat_answer_max_output_tokens // 2


@pytest.mark.parametrize("level", [LOWEST_THINKING_LEVEL, "low"])
async def test_thinking_levels_accepted_by_the_agent_and_grader_models(level):
    settings = get_settings()
    for model in sorted({settings.chat_agent_model, settings.chat_grader_model}):
        llm = ChatGoogleGenerativeAI(
            model=model, temperature=0, thinking_level=level, max_retries=1
        )
        response = await llm.ainvoke("Reply with the single word OK.")
        print(f"\n{model} thinking_level={level}: {response.usage_metadata}")
        assert response.content
