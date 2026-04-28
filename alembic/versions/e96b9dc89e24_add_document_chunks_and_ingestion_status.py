"""Add document chunks and ingestion status

Revision ID: e96b9dc89e24
Revises: e00b14820fd4
Create Date: 2026-04-15 16:39:12.430767

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'e96b9dc89e24'
down_revision: Union[str, Sequence[str], None] = 'e00b14820fd4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('parent_chunk',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('chunk_index', sa.Integer(), nullable=False),
    sa.Column('content', sa.String(), nullable=False),
    sa.Column('document_id', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['document_id'], ['document.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('document_id', 'chunk_index', name='uix_document_chunk_index')
    )
    op.create_table('child_chunk',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('chunk_index', sa.Integer(), nullable=False),
    sa.Column('content', sa.String(), nullable=False),
    sa.Column('embedding', pgvector.sqlalchemy.vector.VECTOR(dim=768), nullable=False),
    sa.Column('parent_id', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['parent_id'], ['parent_chunk.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('parent_id', 'chunk_index', name='uix_parent_chunk_index')
    )
    op.drop_table('documentchunk')
    document_status_enum = postgresql.ENUM('processing', 'completed', 'failed', name='document_status_enum')
    document_status_enum.create(op.get_bind())
    op.add_column(
        'document',
        sa.Column(
            'status',
            document_status_enum,
            nullable=False,
            server_default='processing'
        )
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('document', 'status')
    document_status_enum = postgresql.ENUM('processing', 'completed', 'failed', name='document_status_enum')
    document_status_enum.drop(op.get_bind())
    op.create_table('documentchunk',
    sa.Column('id', sa.INTEGER(), autoincrement=True, nullable=False),
    sa.Column('document_id', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('content', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('embedding', pgvector.sqlalchemy.vector.VECTOR(dim=1536), autoincrement=False, nullable=False),
    sa.Column('chunk_index', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.ForeignKeyConstraint(['document_id'], ['document.id'], name=op.f('documentchunk_document_id_fkey'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('documentchunk_pkey')),
    sa.UniqueConstraint('chunk_index', name=op.f('documentchunk_chunk_index_key'), postgresql_include=[], postgresql_nulls_not_distinct=False)
    )
    op.drop_table('child_chunk')
    op.drop_table('parent_chunk')
