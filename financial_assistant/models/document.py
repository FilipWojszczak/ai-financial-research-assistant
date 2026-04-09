from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, String, func
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
    # when owner_id is None, it means the document is public and can be accessed by any
    #  user
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("user.id"))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    owner: Mapped[User | None] = relationship(back_populates="documents")
    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class DocumentChunk(Base):
    # to not change the existing table name after switching from SQLModel to SQLAlchemy
    __tablename__ = "documentchunk"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("document.id", ondelete="CASCADE")
    )
    content: Mapped[str] = mapped_column(String)

    embedding: Mapped[list[float]] = mapped_column(Vector(1536))
    chunk_index: Mapped[int] = mapped_column(Integer, unique=True)

    document: Mapped[Document] = relationship(back_populates="chunks")
