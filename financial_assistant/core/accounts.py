"""Account deletion that never turns a user's private documents into public ones."""

import secrets
from collections.abc import Sequence
from datetime import timedelta
from uuid import UUID

from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy import delete, exists, func, select, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import ChatThread, ChatTurn, Document, User
from .config import get_settings
from .security import hash_password

# pwdlib raises on a malformed hash, so use a well-formed one whose password nobody
# knows: verify_password() returns False for every input.
TOMBSTONE_PASSWORD_HASH = hash_password(secrets.token_urlsafe(32))
# Bounds every wait for a row lock. The ingestion worker holds one on its document for
# a whole attempt.
LOCK_TIMEOUT = "5s"
# lock_not_available (lock_timeout expired) and deadlock_detected.
_LOCK_ERRORS = {"55P03", "40P01"}


class DocumentsBusyError(Exception):
    """A document of the account is locked by a running ingestion."""


class AccountChangedError(Exception):
    """A document was added to the account while it was being deleted."""


class ChatBusyError(Exception):
    """A chat turn of the account is running."""


class AccountBusyError(Exception):
    """Another request of the account (an upload, a chat) holds it locked."""


def _is_lock_error(exc: DBAPIError) -> bool:
    return getattr(exc.orig, "sqlstate", None) in _LOCK_ERRORS


def tombstone_email(user_id: int) -> str:
    return f"deleted-{user_id}@deleted.invalid"


async def delete_account(
    session: AsyncSession,
    user: User,
    *,
    delete_documents: bool = False,
    checkpointer: BaseCheckpointSaver | None = None,
) -> list[int]:
    """Delete or tombstone the account; return ids of deleted documents.

    Chat data is deleted either way. The caller commits, then removes the returned
    documents' PDFs.
    """
    user_id = user.id
    try:
        deleted_ids, thread_ids = await _delete_or_tombstone(
            session, user, delete_documents
        )
        # Flush here, not at the caller's commit, so these errors are mapped too.
        await session.flush()
        # Last, because deleting checkpoints can't be rolled back.
        await _delete_chat_data(session, user_id, thread_ids, checkpointer)
    except IntegrityError as exc:
        # An upload committed after the ownership check: the FK keeps the user.
        raise AccountChangedError from exc
    except DBAPIError as exc:
        if _is_lock_error(exc):
            raise AccountBusyError from exc
        raise
    return deleted_ids


async def _lock_chat_threads(session: AsyncSession, user_id: int) -> Sequence[UUID]:
    """
    Lock the user, then its threads (the order of claim_turn), so no thread can be
    added and no turn can start. A running turn writes its checkpoint when it ends,
    which would outlive the account.
    """
    await session.execute(select(User.id).where(User.id == user_id).with_for_update())
    result = await session.scalars(
        select(ChatThread.id)
        .where(ChatThread.user_id == user_id)
        .order_by(ChatThread.id)
        .with_for_update()
    )
    thread_ids = result.all()
    stale_after = timedelta(seconds=get_settings().chat_turn_stale_seconds)
    if await session.scalar(
        select(
            exists().where(ChatTurn.user_id == user_id, ChatTurn.active(stale_after))
        )
    ):
        raise ChatBusyError
    return thread_ids


async def _delete_chat_data(
    session: AsyncSession,
    user_id: int,
    thread_ids: Sequence[UUID],
    checkpointer: BaseCheckpointSaver | None,
) -> None:
    # Checkpoints first: they live outside this transaction, so if one can't be
    # deleted the rows are rolled back and the next attempt finishes the job.
    if checkpointer is not None:
        for thread_id in thread_ids:
            await checkpointer.adelete_thread(str(thread_id))
    # Explicit: the cascade from the user row doesn't fire for a tombstoned user.
    await session.execute(delete(ChatTurn).where(ChatTurn.user_id == user_id))
    await session.execute(delete(ChatThread).where(ChatThread.user_id == user_id))


async def _delete_or_tombstone(
    session: AsyncSession, user: User, delete_documents: bool
) -> tuple[list[int], Sequence[UUID]]:
    """Return the ids of the deleted documents and of the chat threads to delete."""
    deleted_ids: list[int] = []
    await session.execute(text(f"SET LOCAL lock_timeout = '{LOCK_TIMEOUT}'"))
    if delete_documents:
        try:
            # DELETE has no NOWAIT: the timeout bounds the wait for the worker's lock.
            result = await session.execute(
                delete(Document)
                .where(Document.owner_id == user.id)
                .returning(Document.id)
            )
        except DBAPIError as exc:
            if _is_lock_error(exc):
                raise DocumentsBusyError from exc
            raise
        deleted_ids = list(result.scalars())

    owns_documents = await session.scalar(
        select(exists().where(Document.owner_id == user.id))
    )
    # After the wait for document locks, so a chat started meanwhile is seen.
    thread_ids = await _lock_chat_threads(session, user.id)
    if not owns_documents:
        await session.delete(user)
        return deleted_ids, thread_ids

    # Keep the row so the documents' owner_id stays valid and matches nobody.
    user.email = tombstone_email(user.id)
    user.hashed_password = TOMBSTONE_PASSWORD_HASH
    user.is_active = False
    user.deleted_at = func.now()  # pyright: ignore[reportAttributeAccessIssue]
    return deleted_ids, thread_ids
