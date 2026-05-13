"""Add BM25 index to child chunk

Revision ID: 95900fffcf36
Revises: e96b9dc89e24
Create Date: 2026-05-12 00:00:00.000000

"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "95900fffcf36"
down_revision: str | Sequence[str] | None = "e96b9dc89e24"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Enable pg_search and create a BM25 index on child_chunk.content."""
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_search;")
    op.execute(
        """
        CREATE INDEX child_chunk_bm25_idx ON child_chunk
        USING bm25 (id, content)
        WITH (key_field = 'id');
        """
    )


def downgrade() -> None:
    """Remove BM25 index and pg_search extension."""
    op.execute("DROP INDEX IF EXISTS child_chunk_bm25_idx;")
    op.execute("DROP EXTENSION IF EXISTS pg_search;")
