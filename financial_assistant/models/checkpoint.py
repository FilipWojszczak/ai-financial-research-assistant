"""AI results kept across ingestion attempts of the same document."""

from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import DateTime, ForeignKey, String, UniqueConstraint, func
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class CheckpointKind(StrEnum):
    GRAPH_EXTRACTION = "graph_extraction"
    COMMUNITY_SUMMARY = "community_summary"


class IngestionCheckpoint(Base):
    """
    One model result (graph extraction of a chunk, community summary) committed
    outside the ingestion transaction, so a retry can reuse it. Rows are deleted
    when the document completes and cascade away with the document.
    """

    __tablename__ = "ingestion_checkpoint"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("document.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[CheckpointKind] = mapped_column(
        SAEnum(
            CheckpointKind,
            name="checkpoint_kind_enum",
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
    )
    # SHA-256 of the model name and the full prompt: a changed prompt never hits.
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    result: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "document_id", "kind", "input_hash", name="uix_ingestion_checkpoint_key"
        ),
    )
