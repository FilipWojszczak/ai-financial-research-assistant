"""Create or migrate the LangGraph checkpointer tables; runs once before the app.

Not an Alembic migration: the saver's own migrations include CREATE INDEX
CONCURRENTLY, which cannot run inside Alembic's transaction, and it tracks its
versions in checkpoint_migrations (alembic/env.py ignores the checkpoint tables).
Not in the API lifespan: several uvicorn workers would race on CREATE TABLE.
Idempotent: applied migrations are skipped.
"""

import asyncio
import logging

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from ..core.config import get_settings
from ..core.db import libpq_url

logger = logging.getLogger(__name__)


async def setup_checkpointer(database_url: str) -> None:
    async with AsyncPostgresSaver.from_conn_string(libpq_url(database_url)) as saver:
        await saver.setup()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(setup_checkpointer(get_settings().database_url))
    logger.info("LangGraph checkpointer tables are up to date")


if __name__ == "__main__":
    main()
