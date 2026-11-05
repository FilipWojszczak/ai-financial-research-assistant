"""Database side of a chat turn: concurrency guard, budget and accounting."""

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from uuid import UUID

from langchain_core.messages.ai import UsageMetadata
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ...models import ChatThread, ChatTurn, Document
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


def thread_title(question: str) -> str:
    return " ".join(question.split())[:_TITLE_CHARS]


def _utc_midnight() -> datetime:
    return datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)


async def tokens_used_today(session: AsyncSession, user_id: int) -> int:
    """Input + output tokens of the user's turns since UTC midnight."""
    used = await session.scalar(
        select(
            func.coalesce(func.sum(ChatTurn.input_tokens + ChatTurn.output_tokens), 0)
        ).where(ChatTurn.user_id == user_id, ChatTurn.started_at >= _utc_midnight())
    )
    return int(used or 0)


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
) -> tuple[ScopeDocument, ...]:
    """Metadata of the documents a turn can search; empty if there are none."""
    rows = await session.execute(
        select(
            Document.id,
            Document.filename,
            Document.company_ticker,
            Document.document_type,
            Document.year,
        )
        .where(Document.id.in_(accessible_documents(scope)))
        .order_by(Document.id)
        .limit(_MAX_SCOPE_DOCUMENTS)
    )
    return tuple(
        ScopeDocument(
            id=row.id,
            filename=row.filename,
            company_ticker=row.company_ticker,
            document_type=row.document_type,
            year=row.year,
        )
        for row in rows
    )
