"""Database side of a chat turn: concurrency guard, budget and accounting."""

import math
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from uuid import UUID

from langchain_core.messages.ai import UsageMetadata
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ...models import ChatThread, ChatTurn, Document, User
from ...models.chat import ChatTurnStatus
from ...retrieval.scope import accessible_documents
from ...retrieval.types import RetrievalScope
from .context import ScopeDocument

_RUNNING_TURN_INDEX = "uix_chat_turn_running_thread"
_TITLE_CHARS = 80
# How many in-scope documents the agent prompt lists.
_MAX_SCOPE_DOCUMENTS = 50


class ThreadBusyError(Exception):
    """Another turn of this thread is running."""


class ThreadGoneError(Exception):
    """The thread was deleted before the turn could be claimed."""


def thread_title(question: str) -> str:
    return " ".join(question.split())[:_TITLE_CHARS]


def _utc_midnight() -> datetime:
    return datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)


def seconds_until_utc_midnight() -> int:
    """When the daily token budget resets."""
    reset = _utc_midnight() + timedelta(days=1)
    return max(1, math.ceil((reset - datetime.now(UTC)).total_seconds()))


async def tokens_used_today(session: AsyncSession, user_id: int) -> int:
    """Input + output tokens of the user's turns since UTC midnight."""
    used = await session.scalar(
        select(
            func.coalesce(func.sum(ChatTurn.input_tokens + ChatTurn.output_tokens), 0)
        ).where(ChatTurn.user_id == user_id, ChatTurn.started_at >= _utc_midnight())
    )
    return int(used or 0)


async def completed_turns(
    session: AsyncSession, *, thread_id: UUID, user_id: int
) -> int:
    # user_id is redundant (the thread's owner), but it is the indexed column.
    count = await session.scalar(
        select(func.count())
        .select_from(ChatTurn)
        .where(
            ChatTurn.user_id == user_id,
            ChatTurn.thread_id == thread_id,
            ChatTurn.status == ChatTurnStatus.COMPLETED,
        )
    )
    return int(count or 0)


async def busy_seconds(
    session: AsyncSession, thread_id: UUID, *, stale_after: timedelta
) -> int | None:
    """
    Seconds until the thread's running turn counts as abandoned at the latest (a
    Retry-After value), or None if no turn is running.
    """
    started_at = await session.scalar(
        select(ChatTurn.started_at).where(
            ChatTurn.thread_id == thread_id, ChatTurn.active(stale_after)
        )
    )
    if started_at is None:
        return None
    remaining = started_at + stale_after - datetime.now(UTC)
    return max(1, math.ceil(remaining.total_seconds()))


async def claim_turn(
    session: AsyncSession,
    *,
    thread_id: UUID,
    user_id: int,
    stale_after: timedelta,
) -> ChatTurn:
    """
    Insert a running turn and commit at once, so no connection is held during the
    turn. A row lock (the ingestion pattern) would pin a pooled connection for the
    whole stream; a crashed API is handled by the stale timeout instead.
    """
    # The locks the insert's foreign keys take anyway, taken first and in the order
    # deletions use (user, thread, turns), so they can't deadlock. If a deletion got
    # there first, this waits for it and finds no row.
    await session.execute(
        select(User.id)
        .where(User.id == user_id)
        .with_for_update(read=True, key_share=True)
    )
    thread = await session.scalar(
        select(ChatThread.id)
        .where(ChatThread.id == thread_id)
        .with_for_update(read=True, key_share=True)
    )
    if thread is None:
        raise ThreadGoneError
    await session.execute(
        update(ChatTurn)
        .where(
            ChatTurn.thread_id == thread_id,
            ChatTurn.status == ChatTurnStatus.RUNNING,
            ChatTurn.started_at < func.now() - stale_after,
        )
        .values(
            status=ChatTurnStatus.FAILED,
            error_code="abandoned",
            finished_at=func.now(),
        )
        .execution_options(synchronize_session=False)
    )
    turn = ChatTurn(thread_id=thread_id, user_id=user_id)
    try:
        # A savepoint: the failed insert must not undo the caller's transaction.
        async with session.begin_nested():
            session.add(turn)
    except IntegrityError as exc:
        constraint = getattr(getattr(exc.orig, "diag", None), "constraint_name", None)
        if constraint == _RUNNING_TURN_INDEX:
            raise ThreadBusyError from exc
        raise
    await session.commit()
    return turn


async def finish_turn(
    session_factory: async_sessionmaker[AsyncSession],
    turn_id: UUID,
    *,
    status: ChatTurnStatus,
    usage: Mapping[str, UsageMetadata],
    error_code: str | None,
    title: str | None = None,
) -> None:
    """Record the turn's outcome; `title` names a thread that has no title yet."""
    async with session_factory() as session:
        # The thread before the turn, the order deletions use.
        turn_thread = select(ChatTurn.thread_id).where(ChatTurn.id == turn_id)
        await session.execute(
            select(ChatThread.id)
            .where(ChatThread.id == turn_thread.scalar_subquery())
            .with_for_update(key_share=True)
        )
        thread_id = await session.scalar(
            update(ChatTurn)
            .where(ChatTurn.id == turn_id)
            .values(
                status=status,
                error_code=error_code,
                input_tokens=sum(item["input_tokens"] for item in usage.values()),
                output_tokens=sum(item["output_tokens"] for item in usage.values()),
                usage=dict(usage),
                finished_at=func.now(),
            )
            .returning(ChatTurn.thread_id)
        )
        if thread_id is not None:
            await session.execute(
                update(ChatThread)
                .where(ChatThread.id == thread_id)
                .values(
                    updated_at=func.now(),
                    title=func.coalesce(ChatThread.title, title),
                )
            )
        await session.commit()


async def load_scope_documents(
    session: AsyncSession, scope: RetrievalScope
) -> tuple[tuple[ScopeDocument, ...], int]:
    """
    Metadata of the documents a turn can search, empty if there are none, and how
    many more are in scope beyond the ones the agent prompt lists.
    """
    result = await session.execute(
        select(
            Document.id,
            Document.filename,
            Document.company_ticker,
            Document.document_type,
            Document.year,
            # A window function is computed before LIMIT: the size of the whole scope.
            func.count().over().label("total"),
        )
        .where(Document.id.in_(accessible_documents(scope)))
        .order_by(Document.id)
        .limit(_MAX_SCOPE_DOCUMENTS)
    )
    rows = result.all()
    documents = tuple(
        ScopeDocument(
            id=row.id,
            filename=row.filename,
            company_ticker=row.company_ticker,
            document_type=row.document_type,
            year=row.year,
        )
        for row in rows
    )
    return documents, rows[0].total - len(rows) if rows else 0
