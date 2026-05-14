"""Add GraphRAG tables: entity, entity_relationship, graph_community, graph_community_membership

Revision ID: f3a2b1c0d9e8
Revises: 95900fffcf36
Create Date: 2026-05-14 00:00:00.000000

"""
from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "f3a2b1c0d9e8"
down_revision: str | Sequence[str] | None = "95900fffcf36"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    entity_type_enum = postgresql.ENUM(
        "company",
        "person",
        "financial_metric",
        "event",
        "product",
        "location",
        "other",
        name="entity_type_enum",
    )
    entity_type_enum.create(op.get_bind())

    op.create_table(
        "entity",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(500), nullable=False),
        sa.Column("type", entity_type_enum, nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "embedding",
            pgvector.sqlalchemy.vector.VECTOR(dim=768),
            nullable=True,
        ),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["document.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_entity_name", "entity", ["name"])

    op.create_table(
        "entity_relationship",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("source_entity_id", sa.Integer(), nullable=False),
        sa.Column("target_entity_id", sa.Integer(), nullable=False),
        sa.Column("relationship_type", sa.String(200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("weight", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("chunk_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["source_entity_id"], ["entity.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["target_entity_id"], ["entity.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["document_id"], ["document.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["chunk_id"], ["parent_chunk.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "graph_community",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("level", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column(
            "embedding",
            pgvector.sqlalchemy.vector.VECTOR(dim=768),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"], ["document.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "graph_community_membership",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("community_id", sa.Integer(), nullable=False),
        sa.Column("entity_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["community_id"], ["graph_community.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["entity_id"], ["entity.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("community_id", "entity_id", name="uix_community_entity"),
    )


def downgrade() -> None:
    op.drop_table("graph_community_membership")
    op.drop_table("graph_community")
    op.drop_table("entity_relationship")
    op.drop_table("entity")
    entity_type_enum = postgresql.ENUM(name="entity_type_enum")
    entity_type_enum.drop(op.get_bind())
