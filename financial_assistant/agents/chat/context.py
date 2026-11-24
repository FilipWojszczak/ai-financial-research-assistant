"""Per-turn runtime context: never checkpointed and never shown to the LLM as tool
arguments, which keeps the user and the thread filters out of its reach."""

from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ...core.config import Settings
from ...models.document import DocumentType
from ...retrieval.hybrid import HybridRetriever
from ...retrieval.types import RetrievalScope


@dataclass(frozen=True)
class ScopeDocument:
    id: int
    filename: str
    company_ticker: str
    document_type: DocumentType
    year: int


class TurnDecision:
    """
    Whether the turn was answered or cancelled, whichever comes first. Once answered,
    only the checkpoint write is left, and interrupting that would leave the history
    and the recorded outcome in disagreement.
    """

    def __init__(self) -> None:
        self._outcome: Literal["answered", "cancelled"] | None = None

    def answer(self) -> bool:
        """False if the turn was already cancelled."""
        self._outcome = self._outcome or "answered"
        return self._outcome == "answered"

    def cancel(self) -> bool:
        """False if the turn was already answered."""
        self._outcome = self._outcome or "cancelled"
        return self._outcome == "cancelled"


@dataclass(frozen=True)
class ChatContext:
    user_id: int
    thread_id: UUID
    turn_id: UUID
    scope: RetrievalScope
    # Metadata of the in-scope documents, listed in the agent prompt.
    scope_documents: tuple[ScopeDocument, ...]
    session_factory: async_sessionmaker[AsyncSession]
    retriever: HybridRetriever
    settings: Settings
    # In-scope documents beyond the ones listed; search still covers them.
    unlisted_documents: int = 0
    decision: TurnDecision = field(default_factory=TurnDecision)
