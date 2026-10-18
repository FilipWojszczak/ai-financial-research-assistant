"""Add user.deleted_at to tombstone deleted accounts that still own documents.

Downgrading while tombstoned users exist drops the marker: their documents stay
inaccessible (owned by an inactive user), but the admin scripts can no longer find them.

Revision ID: 414319e7a35c
Revises: c5d1e8a2f4b7
"""

import sqlalchemy as sa
from alembic import op

revision = "414319e7a35c"
down_revision = "c5d1e8a2f4b7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "user", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("user", "deleted_at")
