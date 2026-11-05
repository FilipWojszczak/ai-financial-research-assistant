"""The chat graph's state, checkpointed once per turn."""

from collections.abc import Sequence
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph.message import add_messages

from ...models.document import DocumentType
from ...retrieval.types import ChunkSource, CommunitySource, Source, source_key
from .models import RetrievalGrade

# Our types stored in checkpoints. langgraph-checkpoint warns about unregistered
# types when loading them and will block them in a future version.
CHAT_STATE_TYPES = (ChunkSource, CommunitySource, RetrievalGrade, DocumentType)


def chat_serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=CHAT_STATE_TYPES)


class NumberedSource(TypedDict):
    # 1-based, what the answer cites as [n].
    ref: int
    source: Source


def merge_sources(
    current: list[NumberedSource], found: Sequence[Source]
) -> list[NumberedSource]:
    """
    Append new sources, numbered in order of first appearance. Refs are assigned
    here rather than in the tool, so they stay deterministic for parallel tool
    calls. Overwrite([]) bypasses this reducer to reset the field.
    """
    merged = list(current)
    seen = {source_key(item["source"]) for item in merged}
    for source in found:
        key = source_key(source)
        if key not in seen:
            seen.add(key)
            merged.append({"ref": len(merged) + 1, "source": source})
    return merged


def merge_dicts[K, V](current: dict[K, V], update: dict[K, V]) -> dict[K, V]:
    return {**current, **update}


class ChatInput(TypedDict):
    question: str


class ChatState(TypedDict):
    # Conversation history: exactly one HumanMessage + one AIMessage per completed turn.
    messages: Annotated[list[AnyMessage], add_messages]
    # Per-turn working memory: agent tool calls, ToolMessages, grader feedback.
    scratchpad: Annotated[list[AnyMessage], add_messages]
    question: str
    sources: Annotated[list[NumberedSource], merge_sources]
    # How many of `sources` have been sent to the client.
    sources_streamed: int
    round: int
    grade: RetrievalGrade | None
    # Sources cited by each answer, keyed by the AIMessage id (= turn id), for history.
    cited_sources: Annotated[dict[str, list[NumberedSource]], merge_dicts]
