"""Add ingestion checkpoints for AI results reused across retries.

Revision ID: c5d1e8a2f4b7
Revises: a4c7d92e810b
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "c5d1e8a2f4b7"
down_revision = "a4c7d92e810b"
branch_labels = None
depends_on = None


def upgrade() -> None:
    checkpoint_kind_enum = postgresql.ENUM(
        "graph_extraction",
        "community_summary",
        name="checkpoint_kind_enum",
        # Created explicitly below; table creation must not emit CREATE TYPE again.
        create_type=False,
    )
    checkpoint_kind_enum.create(op.get_bind())

    op.create_table(
        "ingestion_checkpoint",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("kind", checkpoint_kind_enum, nullable=False),
        sa.Column("input_hash", sa.String(64), nullable=False),
        sa.Column("result", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["document_id"], ["document.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_id", "kind", "input_hash", name="uix_ingestion_checkpoint_key"
        ),
    )


def downgrade() -> None:
    op.drop_table("ingestion_checkpoint")
    checkpoint_kind_enum = postgresql.ENUM(name="checkpoint_kind_enum")
    checkpoint_kind_enum.drop(op.get_bind())
