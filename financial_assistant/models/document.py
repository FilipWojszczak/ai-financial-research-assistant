from datetime import datetime
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, text
from sqlmodel import Field, Relationship, SQLModel

if TYPE_CHECKING:
    from .user import User


class Document(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    filename: str
    company_ticker: str = Field(index=True)
    # document_type: str = Field(nullable=False) TODO: add document type (e.g., 10-K,
    #  10-Q, etc.)
    # year: int = Field(nullable=False) TODO: add year of the document
    owner_id: int | None = Field(default=None, foreign_key="user.id")

    created_at: datetime | None = Field(
        default=None, sa_column_kwargs={"server_default": text("CURRENT_TIMESTAMP")}
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

    embedding: list[float] | None = Field(default=None, sa_column=Column(Vector(1536)))
    chunk_index: int

    document: Document = Relationship(back_populates="chunks")
