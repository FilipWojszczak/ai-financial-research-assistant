"""add_vector_extension

Revision ID: bac3899e0118
Revises:
Create Date: 2026-04-09 10:51:46.856233

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector


# revision identifiers, used by Alembic.
revision: str = 'bac3899e0118'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add the vector extension to the database
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")


def downgrade() -> None:
    # Remove the vector extension from the database
    op.execute("DROP EXTENSION IF EXISTS vector;")
