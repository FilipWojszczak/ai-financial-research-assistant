from typing import Any, cast

import pytest
from langchain_google_genai import ChatGoogleGenerativeAI

from financial_assistant.agents.chat.models import (
    LOWEST_THINKING_LEVEL,
    GeneratedAnswer,
    create_chat_models,
)
from financial_assistant.core.config import Settings

pytestmark = pytest.mark.unit


def test_models_are_built_from_settings():
    settings = Settings(
        secret_key="test",
        algorithm="HS256",
        chat_agent_model="agent-model",
        chat_answer_model="answer-model",
        chat_answer_max_output_tokens=512,
        chat_answer_thinking_level="medium",
        _env_file=None,  # pyright: ignore[reportCallIssue]
    )
    models = create_chat_models(settings)
    agent = models.agent
    assert isinstance(agent, ChatGoogleGenerativeAI)
    assert agent.model == "agent-model"
    assert agent.temperature == 0
    assert agent.thinking_level == LOWEST_THINKING_LEVEL
    assert agent.max_retries == 1
    # with_structured_output wraps the model: RunnableBinding(model) | parser.
    answer = cast(Any, models.answer).first.bound
    assert isinstance(answer, ChatGoogleGenerativeAI)
    assert answer.model == "answer-model"
    assert answer.temperature == settings.chat_answer_temperature
    assert answer.max_output_tokens == 512
    assert answer.thinking_level == "medium"
    assert answer.max_retries == 1


def test_the_answer_text_is_generated_before_its_citations():
    # Structured output follows the schema's property order; the answer streams
    # token by token only if it comes first.
    properties = GeneratedAnswer.model_json_schema()["properties"]
    assert list(properties) == ["answer", "cited_refs"]
