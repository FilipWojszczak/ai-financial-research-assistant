"""Reuse AI results from earlier failed attempts of the same document."""

import hashlib
import logging
from typing import Any

from sqlalchemy import delete, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ..models.checkpoint import CheckpointKind, IngestionCheckpoint

logger = logging.getLogger(__name__)

# The ingestion transaction holds FOR NO KEY UPDATE on the document row, which does
# not block the FOR KEY SHARE lock taken by this table's foreign key. The timeout
# turns any future lock conflict into a skipped checkpoint instead of a hang.
_WRITE_LOCK_TIMEOUT = "5s"


def checkpoint_key(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()


class AttemptCheckpoints:
    """
    Model results committed in their own short transactions, so they survive the
    rollback of a failed ingestion attempt. Best effort: database errors are logged
    and the attempt continues without the checkpoint.
    """

    def __init__(
        self, document_id: int, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        self.document_id = document_id
        self.hits = 0
        self._session_factory = session_factory
        self._results: dict[tuple[CheckpointKind, str], dict[str, Any]] | None = None

    async def _loaded(self) -> dict[tuple[CheckpointKind, str], dict[str, Any]]:
        if self._results is None:
            self._results = {}
            try:
                async with self._session_factory() as session:
                    rows = await session.execute(
                        select(
                            IngestionCheckpoint.kind,
                            IngestionCheckpoint.input_hash,
                            IngestionCheckpoint.result,
                        ).where(IngestionCheckpoint.document_id == self.document_id)
                    )
                    self._results = {(kind, key): res for kind, key, res in rows}
            except Exception:
                logger.warning(
                    "Document %d: cannot read checkpoints",
                    self.document_id,
                    exc_info=True,
                )
        return self._results

    async def get(self, kind: CheckpointKind, key: str) -> dict[str, Any] | None:
        result = (await self._loaded()).get((kind, key))
        if result is not None:
            self.hits += 1
        return result

    async def put(self, kind: CheckpointKind, key: str, result: dict[str, Any]) -> None:
        (await self._loaded())[(kind, key)] = result
        try:
            async with self._session_factory() as session:
                await session.execute(
                    text(f"SET LOCAL lock_timeout = '{_WRITE_LOCK_TIMEOUT}'")
                )
                await session.execute(
                    insert(IngestionCheckpoint)
                    .values(
                        document_id=self.document_id,
                        kind=kind,
                        input_hash=key,
                        result=result,
                    )
                    .on_conflict_do_nothing(constraint="uix_ingestion_checkpoint_key")
                )
                await session.commit()
        except Exception:
            logger.warning(
                "Document %d: cannot store %s checkpoint",
                self.document_id,
                kind,
                exc_info=True,
            )


async def delete_checkpoints(session: AsyncSession, document_id: int) -> None:
    """Drop a document's checkpoints inside the transaction that completes it."""
    await session.execute(
        delete(IngestionCheckpoint).where(
            IngestionCheckpoint.document_id == document_id
        )
    )
