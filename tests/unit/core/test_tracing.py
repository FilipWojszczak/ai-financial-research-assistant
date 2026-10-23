import logging
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from financial_assistant.core import tracing

pytestmark = pytest.mark.unit

_MODULE = "financial_assistant.core.tracing"


def _settings(public_key="pk-lf-test", secret_key="sk-lf-test"):
    return SimpleNamespace(
        langfuse_public_key=public_key,
        langfuse_secret_key=None if secret_key is None else SecretStr(secret_key),
        langfuse_base_url="https://langfuse.example",
        environment="test",
    )


@pytest.fixture(autouse=True)
def fresh_process_state(monkeypatch):
    monkeypatch.setattr(tracing, "_client", None)
    monkeypatch.setattr(tracing, "_client_pid", None)
    monkeypatch.setattr(tracing, "_status_logged_pid", None)


@pytest.fixture
def langfuse():
    with patch(f"{_MODULE}.Langfuse") as langfuse_class:
        yield langfuse_class


def _use_settings(settings):
    return patch(f"{_MODULE}.get_settings", return_value=settings)


@pytest.mark.parametrize(
    ("public_key", "secret_key"),
    [(None, None), ("pk-lf-test", None), (None, "sk-lf-test"), ("", "")],
)
def test_disabled_without_both_keys_constructs_no_client(
    langfuse, caplog, public_key, secret_key
):
    caplog.set_level(logging.INFO, logger=_MODULE)
    with _use_settings(_settings(public_key, secret_key)):
        assert tracing.tracing_enabled() is False
        assert tracing.langchain_callbacks() == []
        with (
            tracing.trace_scope(
                "chat_turn", session_id="s", user_id="u", tags=["chat"], metadata={}
            ),
            tracing.embedding_observation("model", 3),
        ):
            pass
        tracing.record_event("checkpoint_hit", {"kind": "graph_extraction"})
        tracing.shutdown_tracing()
    langfuse.assert_not_called()
    assert caplog.text.count("Langfuse tracing is off") == 1


def test_enabled_client_is_built_from_settings(langfuse, caplog):
    caplog.set_level(logging.INFO, logger=_MODULE)
    with (
        _use_settings(_settings()),
        patch(f"{_MODULE}.CallbackHandler") as handler_class,
    ):
        callbacks = tracing.langchain_callbacks()
        tracing.langchain_callbacks()
    langfuse.assert_called_once_with(
        public_key="pk-lf-test",
        secret_key="sk-lf-test",
        base_url="https://langfuse.example",
        environment="test",
    )
    # A fresh handler per run, bound to the client registered under the same key.
    assert callbacks == [handler_class.return_value]
    assert handler_class.call_count == 2
    handler_class.assert_called_with(public_key="pk-lf-test")
    assert caplog.text.count("Langfuse tracing is on") == 1


def test_trace_scope_propagates_trace_attributes(langfuse):
    client = langfuse.return_value
    client.start_as_current_observation.return_value = nullcontext()
    with (
        _use_settings(_settings()),
        patch(
            f"{_MODULE}.propagate_attributes", return_value=nullcontext()
        ) as propagate,
        tracing.trace_scope(
            "chat_turn",
            session_id="thread-1",
            user_id="7",
            tags=["chat"],
            metadata={"turn_id": "t-1"},
        ),
    ):
        pass
    client.start_as_current_observation.assert_called_once_with(
        name="chat_turn", as_type="span"
    )
    propagate.assert_called_once_with(
        session_id="thread-1", user_id="7", tags=["chat"], metadata={"turn_id": "t-1"}
    )


def test_trace_scope_reraises_errors_from_its_body(langfuse):
    langfuse.return_value.start_as_current_observation.return_value = nullcontext()
    with (
        _use_settings(_settings()),
        patch(f"{_MODULE}.propagate_attributes", return_value=nullcontext()),
        pytest.raises(ValueError, match="boom"),
        tracing.trace_scope("chat_turn", session_id="s", tags=[], metadata={}),
    ):
        raise ValueError("boom")


def test_embedding_observation_and_event_use_the_client(langfuse):
    client = langfuse.return_value
    client.start_as_current_observation.return_value = nullcontext()
    with _use_settings(_settings()):
        with tracing.embedding_observation("models/gemini-embedding-001", 16):
            pass
        tracing.record_event("checkpoint_hit", {"kind": "community_summary"})
    client.start_as_current_observation.assert_called_once_with(
        name="embedding",
        as_type="embedding",
        model="models/gemini-embedding-001",
        metadata={"count": 16},
    )
    client.create_event.assert_called_once_with(
        name="checkpoint_hit", metadata={"kind": "community_summary"}
    )


def test_a_forked_process_builds_its_own_client(langfuse):
    langfuse.side_effect = [MagicMock(name="parent"), MagicMock(name="child")]
    with _use_settings(_settings()):
        with patch(f"{_MODULE}.os.getpid", return_value=100):
            parent = tracing._get_client()
            assert tracing._get_client() is parent
        with patch(f"{_MODULE}.os.getpid", return_value=200):
            child = tracing._get_client()
    assert child is not parent
    assert langfuse.call_count == 2


def test_shutdown_is_idempotent_and_only_stops_this_process_client(langfuse):
    client = langfuse.return_value
    with _use_settings(_settings()):
        tracing._get_client()
        tracing.shutdown_tracing()
        tracing.shutdown_tracing()
        client.shutdown.assert_called_once_with()

        # A client inherited from a prefork parent is dropped, not shut down here.
        with patch(f"{_MODULE}.os.getpid", return_value=100):
            tracing._get_client()
        client.shutdown.reset_mock()
        with patch(f"{_MODULE}.os.getpid", return_value=200):
            tracing.shutdown_tracing()
        client.shutdown.assert_not_called()
    assert tracing._client is None
