"""Seed documents, chunks and communities with controlled vectors."""

import math
from dataclasses import dataclass
from datetime import UTC, datetime

from financial_assistant.core.config import Settings, get_settings
from financial_assistant.models import (
    ChildChunk,
    Document,
    GraphCommunity,
    ParentChunk,
    User,
)
from financial_assistant.models.document import DocumentStatus, DocumentType

DIMENSIONS = 768


def vector(*weights: float) -> list[float]:
    """A unit vector whose leading components are proportional to `weights`."""
    norm = math.sqrt(sum(weight * weight for weight in weights))
    return [weight / norm for weight in weights] + [0.0] * (DIMENSIONS - len(weights))


def axis(index: int) -> list[float]:
    return vector(*([0.0] * index), 1.0)


def retrieval_settings(**overrides) -> Settings:
    return get_settings().model_copy(update=overrides)


@dataclass(frozen=True)
class SeededDocument:
    id: int
    parent_ids: list[int]


class Seeder:
    def __init__(self, factory) -> None:
        self.factory = factory

    async def user(self, email: str, *, tombstoned: bool = False) -> int:
        async with self.factory() as session:
            user = User(email=email, hashed_password="unused")
            if tombstoned:
                user.is_active = False
                user.deleted_at = datetime.now(UTC)
            session.add(user)
            await session.commit()
            return user.id

    async def document(
        self,
        parents: list[list[tuple[str, list[float]]]],
        *,
        owner_id: int | None = None,
        status: DocumentStatus = DocumentStatus.COMPLETED,
        company_ticker: str = "ACME",
        document_type: DocumentType = DocumentType.ANNUAL_REPORT,
        year: int = 2025,
        filename: str = "acme.pdf",
    ) -> SeededDocument:
        """`parents` lists each parent's child chunks as (content, embedding).

        Each parent's content is a plain space-joined concatenation of its children.
        Real ingestion splits children from the parent with overlap, so this is only
        a simplification that preserves "parent contains every child"; don't assert
        on exact parent text or on terms that span chunk boundaries.
        """
        async with self.factory() as session:
            document = Document(
                filename=filename,
                company_ticker=company_ticker,
                document_type=document_type,
                year=year,
                owner_id=owner_id,
                status=status,
            )
            session.add(document)
            await session.flush()
            parent_rows = []
            for parent_index, children in enumerate(parents):
                parent = ParentChunk(
                    chunk_index=parent_index,
                    content=" ".join(content for content, _ in children),
                    document_id=document.id,
                )
                session.add(parent)
                await session.flush()
                parent_rows.append(parent)
                session.add_all(
                    ChildChunk(
                        chunk_index=child_index,
                        content=content,
                        embedding=embedding,
                        parent_id=parent.id,
                    )
                    for child_index, (content, embedding) in enumerate(children)
                )
            await session.commit()
            return SeededDocument(document.id, [parent.id for parent in parent_rows])

    async def community(
        self, document_id: int, title: str, embedding: list[float] | None
    ) -> int:
        async with self.factory() as session:
            community = GraphCommunity(
                document_id=document_id,
                level=0,
                title=title,
                summary=f"Summary of {title}",
                embedding=embedding,
            )
            session.add(community)
            await session.commit()
            return community.id
