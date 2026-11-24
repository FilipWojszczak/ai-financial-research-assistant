from collections.abc import AsyncIterator
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ...agents.chat.history import load_history
from ...agents.chat.runner import ChatTurnRunner
from ...agents.chat.turns import busy_seconds
from ...core.db import get_session
from ...models import ChatThread, Document, User
from ...retrieval.scope import readable_documents
from ...schemas.chat import ChatEvent, ThreadCreate, ThreadDetail, ThreadRead
from ..dependencies.auth import get_current_user
from ..dependencies.chat import (
    find_thread,
    get_chat_runner,
    get_checkpointer,
    open_turn_stream,
    stale_after,
    thread_busy,
)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post(
    "/threads",
    response_model=ThreadRead,
    status_code=201,
    summary="Create a chat thread",
    description=(
        "Creates a conversation, optionally limited to some documents or to one "
        "company. These filters cannot be changed later. Documents that are still "
        "being processed are allowed and become searchable once completed. Returns "
        "404 if a document does not exist or is not accessible."
    ),
)
async def create_thread(
    payload: ThreadCreate,
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
):
    if payload.document_ids is not None:
        readable = await session.scalar(
            select(func.count())
            .select_from(Document)
            .where(Document.id.in_(payload.document_ids), readable_documents(user.id))
        )
        if readable != len(payload.document_ids):
            raise HTTPException(status_code=404, detail="Document not found")
    thread = ChatThread(user_id=user.id, **payload.model_dump())
    session.add(thread)
    try:
        await session.commit()
    except IntegrityError as exc:
        # The only foreign key: the account was deleted meanwhile.
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    await session.refresh(thread)
    return thread


@router.get(
    "/threads",
    response_model=list[ThreadRead],
    summary="List chat threads",
    description="Returns the current user's threads, most recently active first.",
)
async def list_threads(
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0, le=2**31 - 1)] = 0,
):
    result = await session.scalars(
        select(ChatThread)
        .where(ChatThread.user_id == user.id)
        # The id makes pages stable when two threads share a timestamp.
        .order_by(ChatThread.updated_at.desc(), ChatThread.id)
        .limit(limit)
        .offset(offset)
    )
    return result.all()


@router.get(
    "/threads/{thread_id}",
    response_model=ThreadDetail,
    summary="Get a chat thread with its messages",
    description=(
        "Returns a thread of the current user with its questions and answers. "
        "Failed and cancelled turns are not part of the history."
    ),
)
async def get_thread(
    thread_id: UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
    runner: Annotated[ChatTurnRunner, Depends(get_chat_runner)],
):
    thread = await find_thread(session, user, thread_id)
    messages = await load_history(runner.graph, session, thread_id)
    return ThreadDetail(
        **ThreadRead.model_validate(thread).model_dump(), messages=messages
    )


@router.delete(
    "/threads/{thread_id}",
    status_code=204,
    summary="Delete a chat thread",
    description=(
        "Deletes a thread of the current user with its messages. Returns 409 while a "
        "message is still being answered."
    ),
)
async def delete_thread(
    thread_id: UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
    checkpointer: Annotated[BaseCheckpointSaver, Depends(get_checkpointer)],
):
    # Locked, so no turn can start (and later write a checkpoint) meanwhile.
    thread = await find_thread(session, user, thread_id, lock=True)
    retry_after = await busy_seconds(session, thread_id, stale_after=stale_after())
    if retry_after is not None:
        raise thread_busy(retry_after)
    # Checkpoints first: if this fails, the row remains and DELETE can be retried.
    await checkpointer.adelete_thread(str(thread_id))
    # chat_turn rows survive (thread_id becomes NULL) to keep the token budget.
    await session.delete(thread)
    await session.commit()


@router.post(
    "/threads/{thread_id}/messages",
    response_class=EventSourceResponse,
    summary="Send a message and stream the answer",
    description=(
        "Streams Server-Sent Events: `turn`, then `status`, `sources` and `token` "
        "while the answer is produced, and finally `done` or `error`. Disconnecting "
        "cancels the turn.\n\n"
        "Returns 409 with code `thread_busy` while another message of the thread is "
        "being answered or `thread_full` when the thread has too many turns, and 429 "
        "with code `budget_exceeded` when the daily token budget is used up."
    ),
)
async def send_message(
    events: Annotated[AsyncIterator[ChatEvent], Depends(open_turn_stream)],
) -> AsyncIterator[ServerSentEvent]:
    async for event in events:
        # ASCII only: FastAPI splits the data with str.splitlines(), so a raw U+2028
        # or U+0085 (e.g. in a PDF excerpt) would break the JSON apart.
        yield ServerSentEvent(
            event=event.event, raw_data=event.data.model_dump_json(ensure_ascii=True)
        )
