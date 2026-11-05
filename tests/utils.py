from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import asynccontextmanager
from typing import Any, Protocol, cast
from unittest.mock import MagicMock
from uuid import uuid4

from google.genai.errors import ClientError, ServerError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableGenerator, RunnableLambda
from langchain_google_genai._common import GoogleGenerativeAIError
from pydantic import Field
from sqlalchemy.engine import make_url

from financial_assistant.agents.chat.context import ChatContext, ScopeDocument
from financial_assistant.agents.chat.models import ChatModels, RetrievalGrade
from financial_assistant.core.config import Settings
from financial_assistant.models import Document, User
from financial_assistant.models.document import DocumentStatus, DocumentType
from financial_assistant.retrieval.types import (
    ChunkSource,
    CommunitySource,
    RetrievalScope,
)


class UserFactory(Protocol):
    async def __call__(self, email: str, password: str = "securepassword") -> User: ...


class TokenFactory(Protocol):
    def __call__(self, user: User) -> str: ...


class DocumentFactory(Protocol):
    async def __call__(
        self,
        owner_id: int | None = None,
        company_ticker: str = "AAPL",
        document_type: DocumentType = DocumentType.ANNUAL_REPORT,
        year: int = 2023,
        filename: str = "test.pdf",
        status: DocumentStatus = DocumentStatus.COMPLETED,
    ) -> Document: ...


def chunk_source(
    parent_chunk_id: int, *, content: str = "Text", filename: str = "a.pdf"
) -> ChunkSource:
    return ChunkSource(
        parent_chunk_id=parent_chunk_id,
        document_id=1,
        filename=filename,
        company_ticker="ACME",
        document_type=DocumentType.ANNUAL_REPORT,
        year=2025,
        content=content,
        score=0.03,
    )


def community_source(community_id: int) -> CommunitySource:
    return CommunitySource(
        community_id=community_id,
        document_id=1,
        filename="a.pdf",
        company_ticker="ACME",
        title=f"Title {community_id}",
        summary=f"Summary {community_id}",
        similarity=0.8,
    )


class FakeToolCallingChatModel(BaseChatModel):
    """Replays scripted AIMessages (repeating the last) and records every call."""

    responses: list[AIMessage]
    bound_tool_choices: list[Any] = Field(default_factory=list)
    requests: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        self.bound_tool_choices.append(tool_choice)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        index = min(len(self.requests), len(self.responses) - 1)
        self.requests.append(list(messages))
        message = self.responses[index].model_copy(deep=True)
        return ChatResult(generations=[ChatGeneration(message=message)])


def tool_call_message(*queries: str, **arguments: Any) -> AIMessage:
    """The agent's response calling search_documents once per query."""
    return AIMessage(
        "",
        tool_calls=[
            {
                "name": "search_documents",
                "args": {"query": query, **arguments},
                "id": f"call-{uuid4().hex}",
                "type": "tool_call",
            }
            for query in queries
        ],
    )


def fake_chat_models(
    agent: Sequence[AIMessage],
    *,
    grades: Iterable[RetrievalGrade | Exception] = (),
    answer: Sequence[dict[str, Any]] = (),
) -> ChatModels:
    """`grades` are replayed in order; `answer` is the stream of partial dicts."""
    remaining_grades = iter(grades)

    def grade(_messages):
        result = next(remaining_grades, RetrievalGrade(sufficient=True))
        if isinstance(result, Exception):
            raise result
        return result

    async def stream_answer(messages: AsyncIterator[Any]) -> AsyncIterator[dict]:
        async for _ in messages:
            pass
        for partial in answer:
            yield partial

    return ChatModels(
        agent=FakeToolCallingChatModel(responses=list(agent)),
        grader=RunnableLambda(grade),
        answer=RunnableGenerator(stream_answer),
    )


def fake_agent(models: ChatModels) -> FakeToolCallingChatModel:
    return cast(FakeToolCallingChatModel, models.agent)


class FakeHybridRetriever:
    """Returns the scripted sources of each search, repeating the last."""

    def __init__(self, *results: list) -> None:
        self.results = results or ([],)
        self.calls: list[tuple[str, RetrievalScope]] = []

    async def search(self, session, query, scope):
        index = min(len(self.calls), len(self.results) - 1)
        self.calls.append((query, scope))
        return self.results[index]


def chat_settings(**overrides: Any) -> Settings:
    return Settings(
        secret_key="test",
        algorithm="HS256",
        _env_file=None,  # pyright: ignore[reportCallIssue]
        **overrides,
    )


def fake_session_factory() -> Any:
    @asynccontextmanager
    async def factory():
        yield MagicMock()

    return factory


def chat_context(
    retriever: FakeHybridRetriever | None = None,
    *,
    scope: RetrievalScope | None = None,
    **settings: Any,
) -> ChatContext:
    return ChatContext(
        user_id=1,
        thread_id=uuid4(),
        turn_id=uuid4(),
        scope=scope or RetrievalScope(user_id=1),
        scope_documents=(
            ScopeDocument(
                id=1,
                filename="a.pdf",
                company_ticker="ACME",
                document_type=DocumentType.ANNUAL_REPORT,
                year=2025,
            ),
        ),
        session_factory=fake_session_factory(),
        retriever=cast(Any, retriever or FakeHybridRetriever()),
        settings=chat_settings(**settings),
    )


def provider_error(code: int) -> GoogleGenerativeAIError:
    """A Gemini HTTP error wrapped by LangChain the way the SDK integration does it."""
    error_type = ClientError if code < 500 else ServerError
    body = {"error": {"code": code, "message": "test", "status": "TEST"}}
    try:
        try:
            raise error_type(code, body)
        except error_type as sdk_error:
            raise GoogleGenerativeAIError(f"Error ({code})") from sdk_error
    except GoogleGenerativeAIError as wrapped:
        return wrapped


def derive_test_database_url(database_url: str) -> str:
    """Return the same URL pointing at `<database>_test`.

    A URL whose database already ends in `_test` is returned unchanged (idempotent).
    """
    url = make_url(database_url)
    if not url.database:
        raise ValueError("The database URL has no database name.")
    if not url.database.endswith("_test"):
        url = url.set(database=url.database + "_test")
    return url.render_as_string(hide_password=False)
