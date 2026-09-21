"""Publish outbox requests once, or continuously with --loop."""

import argparse
import asyncio
import logging
import math
import signal
from contextlib import suppress

from .core.db import engine
from .core.outbox import publish_pending_documents

logger = logging.getLogger(__name__)


async def run_publisher(
    *, limit: int, poll_interval: float, stop: asyncio.Event
) -> None:
    """Recover from failed passes, and finish the in-flight row before shutdown."""
    if (
        type(limit) is not int
        or limit <= 0
        or not math.isfinite(poll_interval)
        or poll_interval <= 0
    ):
        raise ValueError("limit and poll_interval must be positive and finite")
    while not stop.is_set():
        try:
            count = await publish_pending_documents(limit=limit, stop=stop)
            if count:
                logger.info("Published %d outbox requests", count)
        except Exception as exc:
            # Do not log connection strings or credential-bearing error messages.
            logger.error("Outbox pass failed (%s); retrying", type(exc).__name__)
        if stop.is_set():
            break
        with suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=poll_interval)


async def _main(limit: int, *, continuous: bool, poll_interval: float) -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    try:
        if continuous:
            await run_publisher(limit=limit, poll_interval=poll_interval, stop=stop)
        else:
            count = await publish_pending_documents(limit=limit, stop=stop)
            logger.info("Published %d outbox requests", count)
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--loop", action="store_true", help="Keep polling until stopped"
    )
    parser.add_argument("--poll-interval", type=float, default=2.0)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if not math.isfinite(args.poll_interval) or args.poll_interval <= 0:
        parser.error("--poll-interval must be positive and finite")
    logging.basicConfig(level=logging.INFO)
    asyncio.run(
        _main(args.limit, continuous=args.loop, poll_interval=args.poll_interval)
    )
