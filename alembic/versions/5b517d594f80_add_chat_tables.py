"""Add chat threads and turns for the hybrid RAG chat.

The conversation itself is stored by the LangGraph checkpointer (tables created by
services/setup_checkpointer.py, not by Alembic). Downgrading drops every thread's
metadata and turn accounting; the checkpoints are left behind.

Revision ID: 5b517d594f80
Revises: 414319e7a35c
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "5b517d594f80"
down_revision = "414319e7a35c"
branch_labels = None
depends_on = None


def upgrade() -> None:
    chat_turn_status_enum = postgresql.ENUM(
        "running",
        "completed",
        "failed",
        "cancelled",
        name="chat_turn_status_enum",
        # Created explicitly below; table creation must not emit CREATE TYPE again.
        create_type=False,
    )
    chat_turn_status_enum.create(op.get_bind())

    op.create_table(
        "chat_thread",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(200), nullable=True),
        sa.Column("document_ids", postgresql.ARRAY(sa.Integer()), nullable=True),
        sa.Column("company_ticker", sa.String(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_chat_thread_user_id_updated_at",
        "chat_thread",
        ["user_id", sa.literal_column("updated_at DESC")],
    )

    op.create_table(
        "chat_turn",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("thread_id", sa.Uuid(), nullable=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("status", chat_turn_status_enum, nullable=False),
        sa.Column("error_code", sa.String(50), nullable=True),
        sa.Column("input_tokens", sa.Integer(), server_default="0", nullable=False),
        sa.Column("output_tokens", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "usage",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["thread_id"], ["chat_thread.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_chat_turn_user_id_started_at", "chat_turn", ["user_id", "started_at"]
    )
    # The concurrency guard: at most one running turn per thread.
    op.create_index(
        "uix_chat_turn_running_thread",
        "chat_turn",
        ["thread_id"],
        unique=True,
        postgresql_where=sa.text("status = 'running'"),
    )


def downgrade() -> None:
    op.drop_table("chat_turn")
    op.drop_table("chat_thread")
    chat_turn_status_enum = postgresql.ENUM(name="chat_turn_status_enum")
    chat_turn_status_enum.drop(op.get_bind())
