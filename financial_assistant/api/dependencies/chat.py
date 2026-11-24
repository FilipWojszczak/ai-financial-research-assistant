from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from datetime import timedelta
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, Request
from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...agents.chat.runner import ChatTurnRunner
from ...agents.chat.turns import (
    ThreadBusyError,
    ThreadGoneError,
    busy_seconds,
    claim_turn,
    completed_turns,
    seconds_until_utc_midnight,
    tokens_used_today,
)
from ...core.config import get_settings
from ...core.db import get_session
from ...models import ChatThread, User
from ...schemas.chat import ChatEvent, ErrorEvent, MessageCreate
from .auth import get_current_user


def get_chat_runner(request: Request) -> ChatTurnRunner:
    return request.app.state.chat_runner


def get_checkpointer(request: Request) -> BaseCheckpointSaver:
    return request.app.state.checkpointer


def chat_error(
    status_code: int, code: str, message: str, *, retry_after: int | None = None
) -> HTTPException:
    """An HTTP error whose detail has the shape of the stream's `error` event."""
    return HTTPException(
        status_code=status_code,
        detail=ErrorEvent(code=code, message=message).model_dump(),
        headers=None if retry_after is None else {"Retry-After": str(retry_after)},
    )


def thread_busy(retry_after: int) -> HTTPException:
    return chat_error(
        409,
        "thread_busy",
        "Another message of this thread is still being answered.",
        retry_after=retry_after,
    )


def thread_not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="Thread not found")


def stale_after() -> timedelta:
    return timedelta(seconds=get_settings().chat_turn_stale_seconds)


async def find_thread(
    session: AsyncSession, user: User, thread_id: UUID, *, lock: bool = False
) -> ChatThread:
    """The user's own thread; another user's is a 404, like a missing one."""
    query = select(ChatThread).where(
        ChatThread.id == thread_id, ChatThread.user_id == user.id
    )
    if lock:
        query = query.with_for_update()
    thread = await session.scalar(query)
    if thread is None:
        raise thread_not_found()
    return thread


async def open_turn_stream(
    thread_id: UUID,
    message: MessageCreate,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
    runner: Annotated[ChatTurnRunner, Depends(get_chat_runner)],
) -> AsyncGenerator[AsyncIterator[ChatEvent]]:
    """
    Claim a turn and open its event stream. Everything that can still fail with an
    HTTP status happens here: the body of an SSE endpoint only starts running after
    the response has.
    """
    settings = get_settings()
    thread = await find_thread(session, user, thread_id)
    completed = await completed_turns(session, thread_id=thread_id, user_id=user.id)
    if completed >= settings.chat_max_turns_per_thread:
        raise chat_error(
            409, "thread_full", "This thread is full. Start a new one to continue."
        )
    used = await tokens_used_today(session, user.id)
    if used >= settings.chat_daily_token_budget:
        raise chat_error(
            429,
            "budget_exceeded",
            "The daily token budget is used up. It resets at midnight UTC.",
            retry_after=seconds_until_utc_midnight(),
        )
    try:
        # Commits, so no connection is held while the answer streams.
        turn = await claim_turn(
            session, thread_id=thread_id, user_id=user.id, stale_after=stale_after()
        )
    except ThreadGoneError as exc:
        raise thread_not_found() from exc
    except ThreadBusyError as exc:
        # The other turn may have just ended: retry right away then.
        retry_after = await busy_seconds(session, thread_id, stale_after=stale_after())
        raise thread_busy(retry_after or 1) from exc

    # Closed here, after the response: on a disconnect FastAPI may cancel its
    # producer while that waits for the client, so the cancellation never reaches
    # the stream and the turn would keep running until garbage collection.
    async with aclosing(
        runner.stream(
            user_id=user.id, thread=thread, turn_id=turn.id, question=message.content
        )
    ) as events:
        yield events
