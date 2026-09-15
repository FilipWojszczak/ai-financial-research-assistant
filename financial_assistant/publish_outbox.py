"""One-shot outbox publisher. A supervised recurring service follows in Stage 5."""

import argparse
import asyncio
import logging

from .core.db import engine
from .core.outbox import publish_pending_documents


async def _main(limit: int) -> None:
    try:
        count = await publish_pending_documents(limit=limit)
        logging.getLogger(__name__).info("Published %d outbox requests", count)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_main(args.limit))
