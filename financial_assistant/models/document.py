from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, func
from sqlalchemy import Enum as SAEnum
from sqlmodel import Field, Relationship, SQLModel

if TYPE_CHECKING:
    from .user import User


class DocumentType(StrEnum):
    ANNUAL_REPORT = "10-K"
    QUARTERLY_REPORT = "10-Q"
    CURRENT_REPORT = "8-K"
    EARNINGS_CALL_TRANSCRIPT = "earnings_call_transcript"
    OTHER = "other"


class Document(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    filename: str
    company_ticker: str = Field(index=True)
    document_type: DocumentType = Field(
        sa_column=Column(
            SAEnum(
                DocumentType,
                # Name of the enum type in the database - not necessary but name of type
                #  remains the same in the database even after model updates
                name="document_type_enum",
                values_callable=lambda x: [
                    e.value for e in x
                ],  # Store enum values (instead of names) as strings in the database
            ),
            nullable=False,
        )
    )
    year: int = Field(nullable=False)
    # when owner_id is None, it means the document is public and can be accessed by any
    #  user
    owner_id: int | None = Field(default=None, foreign_key="user.id")

    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
    )

    owner: User = Relationship(back_populates="documents")
    chunks: list["DocumentChunk"] = Relationship(
        back_populates="document",
        sa_relationship_kwargs={
            "cascade": "all, delete-orphan",
            "passive_deletes": True,
        },
    )


class DocumentChunk(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="document.id", ondelete="CASCADE")
    content: str

    embedding: list[float] = Field(sa_column=Column(Vector(1536)))
    chunk_index: int = Field(unique=True)

    document: Document = Relationship(back_populates="chunks")
