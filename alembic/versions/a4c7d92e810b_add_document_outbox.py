"""Add durable document ingestion outbox.

Revision ID: a4c7d92e810b
Revises: f3a2b1c0d9e8
"""

from alembic import op
import sqlalchemy as sa

revision = "a4c7d92e810b"
down_revision = "f3a2b1c0d9e8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "document_outbox",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "next_attempt_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("last_error", sa.String(200), nullable=True),
        sa.ForeignKeyConstraint(["document_id"], ["document.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id"),
    )
    op.create_index(
        "ix_document_outbox_pending",
        "document_outbox",
        ["next_attempt_at", "created_at"],
        postgresql_where=sa.text("published_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_document_outbox_pending", table_name="document_outbox")
    op.drop_table("document_outbox")
