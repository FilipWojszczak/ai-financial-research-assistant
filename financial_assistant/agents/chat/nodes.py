"""Graph nodes: each takes (state, runtime) and returns a partial state update."""

import asyncio
import logging
from collections.abc import Sequence
from functools import partial
from typing import Any, Literal, cast

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from langgraph.types import Overwrite
from pydantic import ValidationError

from ...ai.requests import ai_request
from ...retrieval.types import ChunkSource, CommunitySource
from ...schemas.chat import SOURCE_EXCERPT_CHARS
from .context import ChatContext
from .models import ChatModels, GeneratedAnswer, RetrievalGrade
from .prompts import (
    AGENT_SYSTEM_PROMPT,
    ANSWER_SYSTEM_PROMPT,
    GRADER_PROMPT,
    MISSING_INFORMATION_NOTE,
    REVIEWER_FEEDBACK,
    history_window,
    render_documents,
    render_sources,
)
from .state import ChatState, NumberedSource

logger = logging.getLogger(__name__)


class AnswerUnavailableError(Exception):
    """The model produced no usable answer (e.g. a safety block or truncated JSON)."""


def prepare(state: ChatState) -> dict[str, Any]:
    """Reset the per-turn fields, whatever an earlier failed turn left behind."""
    return {
        "scratchpad": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
        "sources": Overwrite([]),
        "sources_streamed": 0,
        "round": 0,
        "grade": None,
    }


def route_after_agent(state: ChatState) -> Literal["tools", "generate"]:
    last = state["scratchpad"][-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else "generate"


def route_after_grade(
    state: ChatState, runtime: Runtime[ChatContext]
) -> Literal["agent", "generate"]:
    grade = state["grade"]
    settings = runtime.context.settings
    if (grade is not None and grade.sufficient) or (
        state["round"] >= settings.chat_max_retrieval_rounds
    ):
        return "generate"
    return "agent"


def context_sources(
    sources: Sequence[NumberedSource], max_chunks: int
) -> list[NumberedSource]:
    """The first `max_chunks` chunk sources and every community source."""
    chunks = [item for item in sources if isinstance(item["source"], ChunkSource)]
    allowed = {item["ref"] for item in chunks[:max_chunks]}
    return [
        item
        for item in sources
        if item["ref"] in allowed or isinstance(item["source"], CommunitySource)
    ]


def _cited_snapshot(item: NumberedSource) -> NumberedSource:
    """
    What the history keeps of a cited source. Checkpoints store the whole mapping
    again every turn, so only the excerpt the API shows is kept.
    """
    source = item["source"]
    field = "content" if isinstance(source, ChunkSource) else "summary"
    text = getattr(source, field)[:SOURCE_EXCERPT_CHARS]
    return {"ref": item["ref"], "source": source.model_copy(update={field: text})}


class ChatNodes:
    """The nodes that call models."""

    def __init__(self, models: ChatModels, tools: Sequence[BaseTool]) -> None:
        self._models = models
        self._tools = tools

    async def agent(
        self, state: ChatState, runtime: Runtime[ChatContext]
    ) -> dict[str, Any]:
        context = runtime.context
        settings = context.settings
        system_prompt = AGENT_SYSTEM_PROMPT.format(
            documents=render_documents(
                context.scope_documents, context.unlisted_documents
            )
        )
        messages = [
            SystemMessage(system_prompt),
            *history_window(state["messages"], settings.chat_history_turns),
            HumanMessage(state["question"]),
            *state["scratchpad"],
        ]
        model = self._models.agent.bind_tools(
            self._tools, tool_choice="any" if state["round"] == 0 else "auto"
        )
        # Stall retries are safe: the agent's output is never streamed to the client.
        response = cast(
            AIMessage,
            await ai_request(
                partial(model.ainvoke, messages),
                deadline_seconds=settings.chat_agent_timeout_seconds,
            ),
        )
        limit = settings.chat_max_tool_calls_per_round
        if len(response.tool_calls) > limit:
            logger.warning(
                "Turn %s: dropping %d of %d tool calls",
                context.turn_id,
                len(response.tool_calls) - limit,
                len(response.tool_calls),
            )
            response = response.model_copy(
                update={"tool_calls": response.tool_calls[:limit]}
            )
        return {"scratchpad": [response], "round": state["round"] + 1}

    async def grade(
        self, state: ChatState, runtime: Runtime[ChatContext]
    ) -> dict[str, Any]:
        context = runtime.context
        settings = context.settings
        sources = state["sources"]
        # Here, not in the tool: refs are assigned by the reducer after the tools ran.
        new_sources = sources[state["sources_streamed"] :]
        if new_sources:
            runtime.stream_writer({"type": "sources", "sources": new_sources})
        runtime.stream_writer({"type": "status", "stage": "grading"})

        rendered = render_sources(
            context_sources(sources, settings.retrieval_max_context_sources)
        )
        messages = [
            SystemMessage(GRADER_PROMPT),
            HumanMessage(f"Question: {state['question']}\n\n{rendered}"),
        ]
        update: dict[str, Any] = {"sources_streamed": len(sources)}
        try:
            grade = await ai_request(
                partial(self._models.grader.ainvoke, messages),
                deadline_seconds=settings.chat_agent_timeout_seconds,
            )
        except OutputParserException:
            # A broken grade must not fail the turn: the answer prompt already
            # handles missing information.
            logger.warning(
                "Turn %s: grader output was not parseable",
                context.turn_id,
                exc_info=True,
            )
            return {**update, "grade": RetrievalGrade(sufficient=True)}
        update["grade"] = grade
        if not grade.sufficient:
            feedback = REVIEWER_FEEDBACK.format(missing=grade.missing or "unknown.")
            update["scratchpad"] = [HumanMessage(feedback)]
        return update

    async def generate(
        self, state: ChatState, runtime: Runtime[ChatContext]
    ) -> dict[str, Any]:
        context = runtime.context
        settings = context.settings
        write = runtime.stream_writer
        write({"type": "status", "stage": "answering"})

        sources = context_sources(
            state["sources"], settings.retrieval_max_context_sources
        )
        request = f"{render_sources(sources)}\n\nQuestion: {state['question']}"
        grade = state["grade"]
        if grade is not None and not grade.sufficient and grade.missing:
            note = MISSING_INFORMATION_NOTE.format(missing=grade.missing)
            request = f"{request}\n\n{note}"
        messages = [
            SystemMessage(ANSWER_SYSTEM_PROMPT),
            *history_window(state["messages"], settings.chat_history_turns),
            HumanMessage(request),
        ]

        async def stream_answer() -> dict[str, Any] | None:
            streamed = ""
            final = None
            async for partial_answer in self._models.answer.astream(messages):
                final = partial_answer
                text = partial_answer.get("answer", "")
                # Skip a partial that isn't a prefix extension (a parser artefact);
                # the final validated text is authoritative.
                if len(text) > len(streamed) and text.startswith(streamed):
                    write({"type": "token", "text": text[len(streamed) :]})
                    streamed = text
            return final

        # attempts=1: resending after tokens reached the client would duplicate text.
        final = await ai_request(
            stream_answer,
            deadline_seconds=settings.chat_answer_timeout_seconds,
            attempts=1,
        )
        try:
            answer = GeneratedAnswer.model_validate(final)
        except ValidationError as exc:
            raise AnswerUnavailableError from exc
        if not answer.answer.strip():
            raise AnswerUnavailableError
        # No await from here on, so the runner can rely on the decision.
        if not context.decision.answer():
            raise asyncio.CancelledError

        cited_refs = set(answer.cited_refs)
        cited = [item for item in sources if item["ref"] in cited_refs]
        turn_id = str(context.turn_id)
        write(
            {
                "type": "answer",
                "id": turn_id,
                "content": answer.answer,
                "citations": [item["ref"] for item in cited],
                "sources": cited,
            }
        )
        return {
            "messages": [
                HumanMessage(state["question"], id=f"{turn_id}:q"),
                AIMessage(answer.answer, id=turn_id),
            ],
            "cited_sources": {turn_id: [_cited_snapshot(item) for item in cited]},
            "scratchpad": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
        }
