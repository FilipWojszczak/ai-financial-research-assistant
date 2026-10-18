from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

if TYPE_CHECKING:
    from .document import Document


class User(Base):
    # to not change the existing table name after switching from SQLModel to SQLAlchemy
    __tablename__ = "user"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String, index=True, unique=True)
    hashed_password: Mapped[str] = mapped_column(String)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
    # Set when the account is deleted but its documents are retained (tombstone).
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    documents: Mapped[list[Document]] = relationship(
        # passive_deletes="all" stops SQLAlchemy from touching child rows when a user
        # is deleted. Without it, session.delete(user) runs
        # UPDATE document SET owner_id = NULL, and owner_id IS NULL means "public",
        # so every private document of the user would become readable by everyone.
        # With it, deleting a user who still owns documents fails with an
        # IntegrityError from the foreign key. Accounts that own documents are
        # tombstoned instead (see core/accounts.py: delete_account).
        back_populates="owner",
        passive_deletes="all",
    )
