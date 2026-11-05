from dataclasses import dataclass
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel

from ..models.document import DocumentType
from ..retrieval.types import ChunkSource, Source

SOURCE_EXCERPT_CHARS = 300


class SourceRead(BaseModel):
    ref: int
    kind: Literal["chunk", "community"]
    document_id: int
    filename: str
    company_ticker: str
    # Chunk sources only.
    document_type: DocumentType | None = None
    year: int | None = None
    excerpt: str
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
    code: str
    message: str


@dataclass(frozen=True)
class ChatEvent:
    """One Server-Sent Event of a chat turn."""

    event: Literal["turn", "status", "sources", "token", "done", "error"]
    data: BaseModel
