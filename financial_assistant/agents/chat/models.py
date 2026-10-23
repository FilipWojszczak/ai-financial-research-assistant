"""The chat agent's models and the structured outputs they return."""

from dataclasses import dataclass
from typing import Any, Literal, cast

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from ...core.config import Settings

# Gemini 3 models think by default, and thinking tokens are billed as output. The
# agent and grader make small routing decisions, so they think as little as the
# model accepts (confirmed by tests/llm/agents/chat/test_models_smoke.py).
LOWEST_THINKING_LEVEL: Literal["minimal"] = "minimal"


class RetrievalGrade(BaseModel):
    sufficient: bool = Field(
        description="True if the sources contain what is needed to answer"
    )
    missing: str | None = Field(
        default=None,
        description="If not sufficient: the missing information, one sentence",
    )


class GeneratedAnswer(BaseModel):
    # First property, so it is generated (and streamed) before the citations.
    answer: str = Field(description="Markdown answer with inline [n] citations")
    cited_refs: list[int] = Field(
        description="Refs of the sources the answer relies on"
    )


@dataclass(frozen=True)
class ChatModels:
    # Tools are bound in the agent node, with tool_choice depending on the round.
    agent: BaseChatModel
    grader: Runnable[LanguageModelInput, RetrievalGrade]
    # A dict schema makes LangChain parse with JsonOutputParser, which yields
    # growing partial dicts while streaming; validate the last one with
    # GeneratedAnswer.model_validate.
    answer: Runnable[LanguageModelInput, dict[str, Any]]


def create_chat_models(settings: Settings) -> ChatModels:
    # max_retries=1 is a single request (as for the ingestion models): one
    # LangChain retry policy is enough, ai_request handles stalls.
    agent = ChatGoogleGenerativeAI(
        model=settings.chat_agent_model,
        temperature=0,
        thinking_level=LOWEST_THINKING_LEVEL,
        max_retries=1,
    )
    grader = ChatGoogleGenerativeAI(
        model=settings.chat_grader_model,
        temperature=0,
        thinking_level=LOWEST_THINKING_LEVEL,
        max_retries=1,
    )
    answer = ChatGoogleGenerativeAI(
        model=settings.chat_answer_model,
        temperature=settings.chat_answer_temperature,
        max_output_tokens=settings.chat_answer_max_output_tokens,
        thinking_level=settings.chat_answer_thinking_level,
        max_retries=1,
    )
    return ChatModels(
        agent=agent,
        grader=cast(
            Runnable[LanguageModelInput, RetrievalGrade],
            grader.with_structured_output(RetrievalGrade),
        ),
        answer=cast(
            Runnable[LanguageModelInput, dict[str, Any]],
            answer.with_structured_output(
                GeneratedAnswer.model_json_schema(), method="json_schema"
            ),
        ),
    )
