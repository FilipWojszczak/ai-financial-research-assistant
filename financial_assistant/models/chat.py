"""Chat threads and the accounting of their turns.

The conversation itself (messages, cited sources) lives in the LangGraph Postgres
checkpointer under thread_id = chat_thread.id; these tables hold what the API
queries: ownership, filters, concurrency and token usage.
"""

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Uuid,
    func,
    text,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class ChatTurnStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ChatThread(Base):
    __tablename__ = "chat_thread"

    # Also the LangGraph thread_id of the conversation's checkpoints.
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("user.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str | None] = mapped_column(String(200))
    # Filters fixed at creation. No FK: a deleted document simply drops out of the
    # thread's scope.
    document_ids: Mapped[list[int] | None] = mapped_column(ARRAY(Integer))
    company_ticker: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    # Set when a turn finishes; threads are listed newest first.
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index(
            "ix_chat_thread_user_id_updated_at",
            "user_id",
            text("updated_at DESC"),
        ),
    )


class ChatTurn(Base):
    __tablename__ = "chat_turn"

    # Also the id of the turn's answer (AIMessage) in the conversation history.
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # SET NULL, not CASCADE: deleting a thread must not reset the daily token budget.
    thread_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("chat_thread.id", ondelete="SET NULL")
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey("user.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[ChatTurnStatus] = mapped_column(
        SAEnum(
            ChatTurnStatus,
            name="chat_turn_status_enum",
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
        default=ChatTurnStatus.RUNNING,
    )
    # A code only, never a provider message.
    error_code: Mapped[str | None] = mapped_column(String(50))
    # Totals across all chat models of the turn; `usage` holds the per-model
    # UsageMetadata.
    input_tokens: Mapped[int] = mapped_column(Integer, server_default="0")
    output_tokens: Mapped[int] = mapped_column(Integer, server_default="0")
    usage: Mapped[dict[str, Any]] = mapped_column(
        JSONB, server_default=text("'{}'::jsonb")
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        # Serves the daily budget: SUM(tokens) WHERE user_id = ? AND started_at >= ?
        Index("ix_chat_turn_user_id_started_at", "user_id", "started_at"),
        # At most one running turn per thread: the concurrency guard behind 409.
        Index(
            "uix_chat_turn_running_thread",
            "thread_id",
            unique=True,
            postgresql_where=text("status = 'running'"),
        ),
    )
