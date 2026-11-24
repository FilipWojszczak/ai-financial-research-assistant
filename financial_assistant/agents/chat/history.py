"""The conversation as the API shows it, read from the thread's latest checkpoint."""

from uuid import UUID

from langchain_core.messages import AIMessage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...models import ChatTurn
from ...schemas.chat import MessageRead, SourceRead
from .graph import ChatGraph


async def load_history(
    graph: ChatGraph, session: AsyncSession, thread_id: UUID
) -> list[MessageRead]:
    snapshot = await graph.aget_state({"configurable": {"thread_id": str(thread_id)}})
    messages = snapshot.values.get("messages", [])
    cited_sources = snapshot.values.get("cited_sources", {})

    # Set by the generate node: an answer's id is its turn's id, and the question's
    # is "<turn id>:q".
    turn_ids = [UUID(item.id) for item in messages if isinstance(item, AIMessage)]
    result = await session.execute(
        select(ChatTurn.id, ChatTurn.started_at, ChatTurn.finished_at).where(
            ChatTurn.id.in_(turn_ids)
        )
    )
    turns = {str(row.id): row for row in result}

    history: list[MessageRead] = []
    for message in messages:
        message_id: str = message.id
        turn = turns.get(message_id.removesuffix(":q"))
        if isinstance(message, AIMessage):
            history.append(
                MessageRead(
                    id=message_id,
                    role="assistant",
                    content=message.text,
                    sources=[
                        SourceRead.from_source(item["ref"], item["source"])
                        for item in cited_sources.get(message_id, [])
                    ],
                    created_at=turn.finished_at or turn.started_at if turn else None,
                )
            )
        else:
            history.append(
                MessageRead(
                    id=message_id,
                    role="user",
                    content=message.text,
                    created_at=turn.started_at if turn else None,
                )
            )
    return history
