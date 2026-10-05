"""A persistent asyncio loop for each sequential Celery worker process."""

import asyncio
import os

from celery.signals import worker_process_shutdown, worker_shutdown

_runner: asyncio.Runner | None = None
_runner_pid: int | None = None


def get_worker_runner() -> asyncio.Runner:
    """Keep async SDK clients on one loop; initialize lazily after prefork."""
    global _runner, _runner_pid
    pid = os.getpid()
    if _runner is None or _runner_pid != pid:
        _runner = asyncio.Runner()
        _runner_pid = pid
    return _runner


@worker_process_shutdown.connect
@worker_shutdown.connect
def close_worker_runner(**kwargs) -> None:
    """Close the process's loop on graceful worker shutdown."""
    global _runner, _runner_pid
    if _runner is not None and _runner_pid == os.getpid():
        _runner.close()
    _runner = None
    _runner_pid = None
