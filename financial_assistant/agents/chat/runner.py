"""Run one chat turn and turn it into a stream of events for the API."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any
from uuid import UUID

import psycopg
from langchain_core.callbacks import UsageMetadataCallbackHandler
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ...ai.requests import is_permanent_provider_error, is_provider_error
from ...core.config import Settings
from ...core.tracing import langchain_callbacks, trace_scope
from ...models import ChatThread
from ...models.chat import ChatTurnStatus
from ...retrieval.hybrid import HybridRetriever
from ...retrieval.types import RetrievalScope
from ...schemas.chat import (
    AnswerMessage,
    ChatEvent,
    DoneEvent,
    ErrorEvent,
    SourceRead,
    SourcesEvent,
    StatusEvent,
    TokenEvent,
    TokenUsage,
    TurnEvent,
)
from .context import ChatContext
from .graph import ChatGraph
from .nodes import AnswerUnavailableError
from .state import NumberedSource
from .tools import RetrievalUnavailableError
from .turns import finish_turn, load_scope_documents, thread_title

logger = logging.getLogger(__name__)

NO_DOCUMENTS_ANSWER = (
    "No processed documents are available in this chat's scope, so I can't answer "
    "yet. Upload a document or wait until processing completes."
)
_ERROR_MESSAGES = {
    "timeout": "The answer took too long. Please try again.",
    "model_unavailable": "The language model is unavailable. Please try again later.",
    "model_error": "The language model rejected the request.",
    "retrieval_unavailable": "Searching the documents failed. Please try again.",
    "answer_unavailable": "No answer could be generated. Try rephrasing the question.",
    "internal": "Something went wrong. Please try again.",
}

# Turns in progress, so the lifespan can finish them on shutdown.
_running_turns: set[asyncio.Task[None]] = set()


def error_code_for(exc: Exception) -> str:
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, AnswerUnavailableError):
        return "answer_unavailable"
    if is_permanent_provider_error(exc):
        return "model_error"
    if isinstance(exc, RetrievalUnavailableError | SQLAlchemyError | psycopg.Error):
        return "retrieval_unavailable"
    if is_provider_error(exc):
        return "model_unavailable"
    return "internal"


def _cancel_once(task: asyncio.Task[None]) -> None:
    # A second cancellation would interrupt the cleanup of the first one.
    if not task.cancelling():
        task.cancel()


async def cancel_running_turns() -> None:
    """Cancel the running turns and wait until they have recorded their outcome."""
    tasks = list(_running_turns)
    for task in tasks:
        _cancel_once(task)
    await asyncio.gather(*tasks, return_exceptions=True)


def _source_reads(sources: list[NumberedSource]) -> list[SourceRead]:
    return [SourceRead.from_source(item["ref"], item["source"]) for item in sources]


def _event_from_stream(payload: dict[str, Any]) -> ChatEvent | None:
    """The client event of a custom stream payload; the answer is sent as `done`."""
    match payload["type"]:
        case "status":
            return ChatEvent(
                "status",
                StatusEvent(stage=payload["stage"], query=payload.get("query")),
            )
        case "sources":
            return ChatEvent(
                "sources", SourcesEvent(sources=_source_reads(payload["sources"]))
            )
        case "token":
            return ChatEvent("token", TokenEvent(text=payload["text"]))
    return None


class ChatTurnRunner:
    def __init__(
        self,
        graph: ChatGraph,
        session_factory: async_sessionmaker[AsyncSession],
        retriever: HybridRetriever,
        settings: Settings,
    ) -> None:
        self.graph = graph
        self._session_factory = session_factory
        self._retriever = retriever
        self._settings = settings

    async def stream(
        self, *, user_id: int, thread: ChatThread, turn_id: UUID, question: str
    ) -> AsyncGenerator[ChatEvent]:
        """
        Run the turn in its own task and yield its events. FastAPI's SSE producer
        runs this generator in anyio cancel scopes, which re-raise cancellation at
        every await, so cleanup awaits here would be cancelled again. A plain task
        gets a single CancelledError and can finish recording the turn.
        """
        queue: asyncio.Queue[ChatEvent | None] = asyncio.Queue()
        task = asyncio.create_task(
            self._run(
                queue,
                user_id=user_id,
                thread=thread,
                turn_id=turn_id,
                question=question,
            )
        )
        _running_turns.add(task)
        task.add_done_callback(_running_turns.discard)
        try:
            while (event := await queue.get()) is not None:
                yield event
        finally:
            # Client disconnect or server shutdown; synchronous, so nothing can
            # interrupt it.
            _cancel_once(task)

    async def _run(
        self,
        queue: asyncio.Queue[ChatEvent | None],
        *,
        user_id: int,
        thread: ChatThread,
        turn_id: UUID,
        question: str,
    ) -> None:
        usage = UsageMetadataCallbackHandler()

        async def record(status: ChatTurnStatus, error_code: str | None) -> bool:
            try:
                await finish_turn(
                    self._session_factory,
                    turn_id,
                    status=status,
                    usage=usage.usage_metadata,
                    error_code=error_code,
                    title=thread_title(question),
                )
            except Exception:
                # The turn stays `running` until it counts as stale.
                logger.exception("Turn %s: cannot record its outcome", turn_id)
                return False
            logger.info(
                "Turn %s %s: usage %s", turn_id, status.value, usage.usage_metadata
            )
            return True

        def fail(code: str) -> None:
            message = _ERROR_MESSAGES[code]
            queue.put_nowait(ChatEvent("error", ErrorEvent(code=code, message=message)))

        queue.put_nowait(ChatEvent("turn", TurnEvent(turn_id=turn_id)))
        try:
            with trace_scope(
                "chat_turn",
                session_id=str(thread.id),
                user_id=str(user_id),
                tags=["chat"],
                metadata={"turn_id": str(turn_id)},
            ):
                async with asyncio.timeout(self._settings.chat_turn_timeout_seconds):
                    answer = await self._execute(
                        queue,
                        usage,
                        user_id=user_id,
                        thread=thread,
                        turn_id=turn_id,
                        question=question,
                    )
        except asyncio.CancelledError:
            await record(ChatTurnStatus.CANCELLED, None)
            raise
        except Exception as exc:
            code = error_code_for(exc)
            if code == "internal":
                logger.exception("Turn %s failed", turn_id)
            else:
                logger.warning(
                    "Turn %s failed: %s (%s)", turn_id, code, type(exc).__name__
                )
            await record(ChatTurnStatus.FAILED, code)
            fail(code)
        else:
            # Recorded first: a client may send the next message right after `done`.
            if await record(ChatTurnStatus.COMPLETED, None):
                queue.put_nowait(
                    ChatEvent("done", self._done_event(turn_id, answer, usage))
                )
            else:
                fail("internal")
        finally:
            queue.put_nowait(None)

    async def _execute(
        self,
        queue: asyncio.Queue[ChatEvent | None],
        usage: UsageMetadataCallbackHandler,
        *,
        user_id: int,
        thread: ChatThread,
        turn_id: UUID,
        question: str,
    ) -> dict[str, Any]:
        settings = self._settings
        document_ids = thread.document_ids
        scope = RetrievalScope(
            user_id=user_id,
            document_ids=None if document_ids is None else tuple(document_ids),
            company_ticker=thread.company_ticker,
        )
        async with self._session_factory() as session:
            scope_documents = await load_scope_documents(session, scope)
        if not scope_documents:
            # No model calls and no checkpoint: there is nothing to search.
            queue.put_nowait(ChatEvent("token", TokenEvent(text=NO_DOCUMENTS_ANSWER)))
            return {
                "id": str(turn_id),
                "content": NO_DOCUMENTS_ANSWER,
                "citations": [],
                "sources": [],
            }

        context = ChatContext(
            user_id=user_id,
            thread_id=thread.id,
            turn_id=turn_id,
            scope=scope,
            scope_documents=scope_documents,
            session_factory=self._session_factory,
            retriever=self._retriever,
            settings=settings,
        )
        answer: dict[str, Any] | None = None
        # Every client event comes from the custom stream, so the agent's and
        # grader's tokens are never sent. durability="exit" writes the state once, at
        # the end: `messages` only changes in the last node, so a failed or cancelled
        # turn never leaves half a turn in the history.
        async for payload in self.graph.astream(
            {"question": question},
            config={
                "configurable": {"thread_id": str(thread.id)},
                "callbacks": [usage, *langchain_callbacks()],
                # Backstop; the loop itself ends through `round`.
                "recursion_limit": 3 * settings.chat_max_retrieval_rounds + 4,
                "run_name": "chat_turn",
            },
            context=context,
            stream_mode="custom",
            durability="exit",
        ):
            if payload["type"] == "answer":
                answer = payload
            elif (event := _event_from_stream(payload)) is not None:
                queue.put_nowait(event)
        if answer is None:
            raise AnswerUnavailableError
        return answer

    @staticmethod
    def _done_event(
        turn_id: UUID, answer: dict[str, Any], usage: UsageMetadataCallbackHandler
    ) -> DoneEvent:
        totals = usage.usage_metadata.values()
        return DoneEvent(
            turn_id=turn_id,
            message=AnswerMessage(
                id=answer["id"],
                content=answer["content"],
                citations=answer["citations"],
                sources=_source_reads(answer["sources"]),
            ),
            usage=TokenUsage(
                input_tokens=sum(item["input_tokens"] for item in totals),
                output_tokens=sum(item["output_tokens"] for item in totals),
            ),
        )
