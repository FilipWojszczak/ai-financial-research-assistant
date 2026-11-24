from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from ..core.config import get_settings
from ..models.document import DocumentType
from ..retrieval.types import ChunkSource, Source

SOURCE_EXCERPT_CHARS = 300
# PostgreSQL text can't store NUL characters.
_NO_NUL = r"^[^\x00]*$"
MAX_THREAD_DOCUMENTS = 50


class ThreadCreate(BaseModel):
    title: (
        Annotated[
            str,
            StringConstraints(
                strip_whitespace=True, min_length=1, max_length=200, pattern=_NO_NUL
            ),
        ]
        | None
    ) = Field(default=None, description="Default: taken from the first question")
    # The bounds of the INTEGER column: PostgreSQL raises on anything larger.
    document_ids: list[Annotated[int, Field(ge=1, le=2**31 - 1)]] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_THREAD_DOCUMENTS,
        description="Only search these documents",
    )
    company_ticker: (
        Annotated[
            str,
            StringConstraints(strip_whitespace=True, min_length=1, pattern=_NO_NUL),
        ]
        | None
    ) = Field(default=None, description="Only search documents of this company")

    model_config = ConfigDict(extra="forbid")

    @field_validator("document_ids")
    @classmethod
    def _unique_document_ids(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and len(set(value)) != len(value):
            raise ValueError("document_ids must be unique")
        return value


class MessageCreate(BaseModel):
    content: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, pattern=_NO_NUL)
    ]

    model_config = ConfigDict(extra="forbid")

    @field_validator("content")
    @classmethod
    def _bounded_content(cls, value: str) -> str:
        limit = get_settings().chat_max_question_chars
        if len(value) > limit:
            raise ValueError(f"content must be at most {limit} characters")
        return value


class ThreadRead(BaseModel):
    id: UUID
    title: str | None
    document_ids: list[int] | None
    company_ticker: str | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SourceRead(BaseModel):
    ref: int
    kind: Literal["chunk", "community"]
    document_id: int
    filename: str
    company_ticker: str
    # The first 300 characters of the passage (chunk) or the summary (community).
    excerpt: str
    # Chunk sources only.
    document_type: DocumentType | None = None
    year: int | None = None
    # Community sources only.
    title: str | None = None

    @classmethod
    def from_source(cls, ref: int, source: Source) -> Self:
        common = {
            "ref": ref,
            "kind": source.kind,
            "document_id": source.document_id,
            "filename": source.filename,
            "company_ticker": source.company_ticker,
        }
        if isinstance(source, ChunkSource):
            return cls(
                **common,
                document_type=source.document_type,
                year=source.year,
                excerpt=source.content[:SOURCE_EXCERPT_CHARS],
            )
        return cls(
            **common,
            title=source.title,
            excerpt=source.summary[:SOURCE_EXCERPT_CHARS],
        )


class MessageRead(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    # The sources an answer cites; None for questions.
    sources: list[SourceRead] | None = None
    created_at: datetime | None


class ThreadDetail(ThreadRead):
    messages: list[MessageRead]


class TurnEvent(BaseModel):
    turn_id: UUID


class StatusEvent(BaseModel):
    stage: Literal["searching", "grading", "answering"]
    query: str | None = None


class SourcesEvent(BaseModel):
    sources: list[SourceRead]


class TokenEvent(BaseModel):
    text: str


class AnswerMessage(BaseModel):
    id: str
    content: str
    citations: list[int]
    sources: list[SourceRead]


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class DoneEvent(BaseModel):
    turn_id: UUID
    message: AnswerMessage
    usage: TokenUsage


class ErrorEvent(BaseModel):
    """Also the `detail` of the chat endpoints' 409 and 429 responses."""

    code: str
    message: str


@dataclass(frozen=True)
class ChatEvent:
    """One Server-Sent Event of a chat turn."""

    event: Literal["turn", "status", "sources", "token", "done", "error"]
    data: BaseModel
