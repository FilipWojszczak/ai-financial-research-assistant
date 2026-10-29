"""Add an HNSW index for dense retrieval over child chunk embeddings.

vector_cosine_ops: dense queries must order by `<=>` to use it. A plain CREATE INDEX
takes a SHARE lock and blocks ingestion commits into child_chunk while it builds,
which is under a second at ~4.4k rows. At a large scale, build it CONCURRENTLY
inside op.get_context().autocommit_block() instead. No data changes.

Revision ID: ba807050d3e4
Revises: 5b517d594f80
"""

from alembic import op

revision = "ba807050d3e4"
down_revision = "5b517d594f80"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_child_chunk_embedding",
        "child_chunk",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_child_chunk_embedding")
