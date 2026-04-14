from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

if TYPE_CHECKING:
    from .user import User


class DocumentType(StrEnum):
    ANNUAL_REPORT = "10-K"
    QUARTERLY_REPORT = "10-Q"
    CURRENT_REPORT = "8-K"
    EARNINGS_CALL_TRANSCRIPT = "earnings_call_transcript"
    OTHER = "other"


class DocumentStatus(StrEnum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(Base):
    # to not change the existing table name after switching from SQLModel to SQLAlchemy
    __tablename__ = "document"

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String)
    company_ticker: Mapped[str] = mapped_column(String, index=True)
    document_type: Mapped[DocumentType] = mapped_column(
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
    year: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    status: Mapped[DocumentStatus] = mapped_column(
        SAEnum(
            DocumentStatus,
            name="document_status_enum",
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
        default=DocumentStatus.PROCESSING,
    )

    # when owner_id is None, it means the document is public and can be accessed by any
    #  user
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("user.id"))

    owner: Mapped[User | None] = relationship(back_populates="documents")
    parent_chunks: Mapped[list[ParentChunk]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class ParentChunk(Base):
    id: Mapped[int] = mapped_column(primary_key=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(String)

    document_id: Mapped[int] = mapped_column(
        ForeignKey("document.id", ondelete="CASCADE")
    )
    document: Mapped[Document] = relationship(back_populates="parent_chunks")
    child_chunks: Mapped[list[ChildChunk]] = relationship(
        back_populates="parent_chunk",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    # Ensure that the combination of document_id and chunk_index is unique to maintain
    # the order of parent chunks within each document
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uix_document_chunk_index"),
    )


class ChildChunk(Base):
    id: Mapped[int] = mapped_column(primary_key=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(String)
    embedding: Mapped[list[float]] = mapped_column(Vector(768))

    parent_id: Mapped[int] = mapped_column(
        ForeignKey("parent_chunk.id", ondelete="CASCADE")
    )
    parent_chunk: Mapped[ParentChunk] = relationship(back_populates="child_chunks")

    # Ensure that the combination of parent_id and chunk_index is unique to maintain the
    # order of child chunks within each parent chunk
    __table_args__ = (
        UniqueConstraint("parent_id", "chunk_index", name="uix_parent_chunk_index"),
    )
