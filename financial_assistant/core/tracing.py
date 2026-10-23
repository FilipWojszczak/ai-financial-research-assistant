"""Langfuse tracing for chat turns and ingestion attempts; a no-op without keys."""

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager

from langchain_core.callbacks import BaseCallbackHandler
from langfuse import Langfuse, propagate_attributes
from langfuse.langchain import CallbackHandler

from .config import get_settings

logger = logging.getLogger(__name__)

_client: Langfuse | None = None
_client_pid: int | None = None
_status_logged_pid: int | None = None


def tracing_enabled() -> bool:
    settings = get_settings()
    # .env.example ships both keys blank, which pydantic reads as "", not None.
    return bool(
        settings.langfuse_public_key
        and settings.langfuse_secret_key
        and settings.langfuse_secret_key.get_secret_value()
    )


def _get_client() -> Langfuse | None:
    """
    The process's client, created lazily where it is used: Celery imports task
    modules in the parent and then forks, and the export thread does not survive
    a fork. Keyed by PID, like tasks/async_runner.get_worker_runner.
    """
    global _client, _client_pid, _status_logged_pid
    pid = os.getpid()
    enabled = tracing_enabled()
    if _status_logged_pid != pid:
        _status_logged_pid = pid
        logger.info("Langfuse tracing is %s", "on" if enabled else "off")
    settings = get_settings()
    if not enabled or settings.langfuse_secret_key is None:
        return None
    if _client is None or _client_pid != pid:
        # Built from Settings, not os.environ: .env reaches the app through
        # pydantic-settings only.
        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            base_url=settings.langfuse_base_url,
            environment=settings.environment,
        )
        _client_pid = pid
    return _client


def langchain_callbacks() -> list[BaseCallbackHandler]:
    """A fresh Langfuse handler for one LangChain/LangGraph run, or none."""
    if _get_client() is None:
        return []
    # The handler looks up the client registered under this key, never os.environ.
    return [CallbackHandler(public_key=get_settings().langfuse_public_key)]


@contextmanager
def trace_scope(
    name: str,
    *,
    session_id: str,
    user_id: str | None = None,
    tags: list[str],
    metadata: dict[str, str],
) -> Generator[None]:
    """
    Root span whose session, user, tags and metadata propagate to every observation
    created inside it (Langfuse SDK v4: ids and metadata values ≤ 200 characters).
    """
    client = _get_client()
    if client is None:
        yield
        return
    with (
        client.start_as_current_observation(name=name, as_type="span"),
        propagate_attributes(
            session_id=session_id, user_id=user_id, tags=tags, metadata=metadata
        ),
    ):
        yield


@contextmanager
def embedding_observation(model: str, count: int) -> Generator[None]:
    """LangChain embeddings emit no callbacks, so each provider call gets a span."""
    client = _get_client()
    if client is None:
        yield
        return
    with client.start_as_current_observation(
        name="embedding", as_type="embedding", model=model, metadata={"count": count}
    ):
        yield


def record_event(name: str, metadata: dict[str, str]) -> None:
    client = _get_client()
    if client is not None:
        client.create_event(name=name, metadata=metadata)


def shutdown_tracing() -> None:
    """Flush and stop this process's client; safe to call more than once."""
    global _client, _client_pid
    if _client is not None and _client_pid == os.getpid():
        _client.shutdown()
    _client = None
    _client_pid = None
