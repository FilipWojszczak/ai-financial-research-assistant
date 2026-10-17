"""Account deletion that never turns a user's private documents into public ones."""

import secrets

from sqlalchemy import delete, exists, func, select, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Document, User
from .security import hash_password

# pwdlib raises on a malformed hash, so use a well-formed one whose password nobody
# knows: verify_password() returns False for every input.
TOMBSTONE_PASSWORD_HASH = hash_password(secrets.token_urlsafe(32))
# The ingestion worker holds a row lock on its document for a whole attempt.
DOCUMENT_LOCK_TIMEOUT = "5s"


class DocumentsBusyError(Exception):
    """A document of the account is locked by a running ingestion."""


class AccountChangedError(Exception):
    """A document was added to the account while it was being deleted."""


def tombstone_email(user_id: int) -> str:
    return f"deleted-{user_id}@deleted.invalid"


async def delete_account(
    session: AsyncSession, user: User, *, delete_documents: bool = False
) -> list[int]:
    """Delete or tombstone the account; return ids of deleted documents.

    The caller commits, then removes the returned documents' PDFs.
    """
    try:
        deleted_ids = await _delete_or_tombstone(session, user, delete_documents)
        # Flush here, not at the caller's commit, so these errors are mapped too.
        await session.flush()
    except IntegrityError as exc:
        # An upload committed after the ownership check: the FK keeps the user.
        raise AccountChangedError from exc
    except DBAPIError as exc:
        # PostgreSQL 55P03 (lock_not_available): lock_timeout expired.
        if getattr(exc.orig, "sqlstate", None) == "55P03":
            raise DocumentsBusyError from exc
        raise
    return deleted_ids


async def _delete_or_tombstone(
    session: AsyncSession, user: User, delete_documents: bool
) -> list[int]:
    deleted_ids: list[int] = []
    if delete_documents:
        # DELETE has no NOWAIT: bound the wait for the worker's lock instead.
        await session.execute(
            text(f"SET LOCAL lock_timeout = '{DOCUMENT_LOCK_TIMEOUT}'")
        )
        result = await session.execute(
            delete(Document).where(Document.owner_id == user.id).returning(Document.id)
        )
        deleted_ids = list(result.scalars())

    owns_documents = await session.scalar(
        select(exists().where(Document.owner_id == user.id))
    )
    if not owns_documents:
        await session.delete(user)
        return deleted_ids

    # Keep the row so the documents' owner_id stays valid and matches nobody.
    user.email = tombstone_email(user.id)
    user.hashed_password = TOMBSTONE_PASSWORD_HASH
    user.is_active = False
    user.deleted_at = func.now()  # pyright: ignore[reportAttributeAccessIssue]
    return deleted_ids
