from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import Enum as SAEnum
from sqlalchemy import Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

if TYPE_CHECKING:
    from .document import Document, ParentChunk


class EntityType(StrEnum):
    COMPANY = "company"
    PERSON = "person"
    FINANCIAL_METRIC = "financial_metric"
    EVENT = "event"
    PRODUCT = "product"
    LOCATION = "location"
    OTHER = "other"


class Entity(Base):
    __tablename__ = "entity"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(500), index=True)
    type: Mapped[EntityType] = mapped_column(
        SAEnum(
            EntityType,
            name="entity_type_enum",
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
    )
    description: Mapped[str | None] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(768))

    document_id: Mapped[int] = mapped_column(
        ForeignKey("document.id", ondelete="CASCADE")
    )

    document: Mapped[Document] = relationship(back_populates="entities")
    source_relationships: Mapped[list[EntityRelationship]] = relationship(
        foreign_keys="EntityRelationship.source_entity_id",
        back_populates="source_entity",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    target_relationships: Mapped[list[EntityRelationship]] = relationship(
        foreign_keys="EntityRelationship.target_entity_id",
        back_populates="target_entity",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    community_memberships: Mapped[list[GraphCommunityMembership]] = relationship(
        back_populates="entity",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class EntityRelationship(Base):
    __tablename__ = "entity_relationship"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_entity_id: Mapped[int] = mapped_column(
        ForeignKey("entity.id", ondelete="CASCADE")
    )
    target_entity_id: Mapped[int] = mapped_column(
        ForeignKey("entity.id", ondelete="CASCADE")
    )
    relationship_type: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text)
    weight: Mapped[float] = mapped_column(Float, default=1.0)

    document_id: Mapped[int] = mapped_column(
        ForeignKey("document.id", ondelete="CASCADE")
    )
    chunk_id: Mapped[int | None] = mapped_column(
        ForeignKey("parent_chunk.id", ondelete="SET NULL")
    )

    source_entity: Mapped[Entity] = relationship(
        foreign_keys=[source_entity_id],
        back_populates="source_relationships",
    )
    target_entity: Mapped[Entity] = relationship(
        foreign_keys=[target_entity_id],
        back_populates="target_relationships",
    )
    document: Mapped[Document] = relationship()
    chunk: Mapped[ParentChunk | None] = relationship()


class GraphCommunity(Base):
    __tablename__ = "graph_community"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("document.id", ondelete="CASCADE")
    )
    level: Mapped[int] = mapped_column(Integer, default=0)
    title: Mapped[str] = mapped_column(String(500))
    summary: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(768))

    document: Mapped[Document] = relationship(back_populates="communities")
    members: Mapped[list[GraphCommunityMembership]] = relationship(
        back_populates="community",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class GraphCommunityMembership(Base):
    __tablename__ = "graph_community_membership"

    id: Mapped[int] = mapped_column(primary_key=True)
    community_id: Mapped[int] = mapped_column(
        ForeignKey("graph_community.id", ondelete="CASCADE")
    )
    entity_id: Mapped[int] = mapped_column(ForeignKey("entity.id", ondelete="CASCADE"))

    community: Mapped[GraphCommunity] = relationship(back_populates="members")
    entity: Mapped[Entity] = relationship(back_populates="community_memberships")

    __table_args__ = (
        UniqueConstraint("community_id", "entity_id", name="uix_community_entity"),
    )
