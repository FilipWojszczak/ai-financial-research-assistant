import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tests.utils import provider_error

from financial_assistant.ai.requests import (
    ai_request,
    embed_texts,
    is_permanent_provider_error,
)


def _settings(llm=1.0, embedding=1.0, attempts=3):
    return SimpleNamespace(
        ai_request_timeout_seconds=llm,
        embedding_request_timeout_seconds=embedding,
        ai_request_attempts=attempts,
    )


@pytest.fixture(autouse=True)
def no_retry_pause():
    with patch("financial_assistant.ai.requests._STALL_RETRY_DELAY_SECONDS", 0):
        yield


async def test_ai_deadline_cancels_each_pending_call_then_gives_up():
    cancelled = []

    async def hang():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    with (
        patch(
            "financial_assistant.ai.requests.get_settings",
            return_value=_settings(llm=0.01, attempts=3),
        ),
        pytest.raises(TimeoutError),
    ):
        await ai_request(hang)
    assert len(cancelled) == 3


async def test_stalled_call_is_sent_again_and_its_result_used(caplog):
    calls = 0

    async def stall_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.Event().wait()
        return "answer"

    with patch(
        "financial_assistant.ai.requests.get_settings",
        return_value=_settings(llm=0.01),
    ):
        assert await ai_request(stall_once) == "answer"
    assert calls == 2
    assert "no response within" in caplog.text


async def test_other_errors_are_not_retried_by_the_call_wrapper():
    call = AsyncMock(side_effect=ConnectionError("down"))
    with (
        patch("financial_assistant.ai.requests.get_settings", return_value=_settings()),
        pytest.raises(ConnectionError),
    ):
        await ai_request(call)
    assert call.await_count == 1


async def test_explicit_deadline_overrides_the_llm_default():
    async def slow():
        await asyncio.sleep(0.05)
        return "late"

    with (
        patch(
            "financial_assistant.ai.requests.get_settings",
            return_value=_settings(llm=1.0, attempts=1),
        ),
        pytest.raises(TimeoutError),
    ):
        await ai_request(slow, deadline_seconds=0.01)


async def test_embedding_deadline_applies_to_each_batch_not_total_work():
    async def embed(batch):
        await asyncio.sleep(0.04)
        return [[float(value)] for value in batch]

    model = SimpleNamespace(aembed_documents=AsyncMock(side_effect=embed))
    texts = [str(i) for i in range(65)]
    with patch(
        "financial_assistant.ai.requests.get_settings",
        return_value=_settings(llm=0.01, embedding=0.15),
    ):
        result = await embed_texts(model, texts)
    assert result == [[float(i)] for i in range(65)]
    assert model.aembed_documents.await_count == 5
    assert all(
        len(call.args[0]) <= 16 for call in model.aembed_documents.await_args_list
    )


async def test_stalled_embedding_batch_is_resent_without_redoing_earlier_batches():
    stalled = False

    async def embed(batch):
        nonlocal stalled
        if batch[0] == "16" and not stalled:
            stalled = True
            await asyncio.Event().wait()
        return [[float(value)] for value in batch]

    model = SimpleNamespace(aembed_documents=AsyncMock(side_effect=embed))
    texts = [str(i) for i in range(40)]
    with patch(
        "financial_assistant.ai.requests.get_settings",
        return_value=_settings(embedding=0.01),
    ):
        result = await embed_texts(model, texts)
    assert result == [[float(i)] for i in range(40)]
    # Three batches plus one resend of the stalled second batch.
    assert model.aembed_documents.await_count == 4


async def test_embed_texts_rejects_blank_texts_before_calling_the_model():
    model = MagicMock()
    model.aembed_documents = AsyncMock()

    with pytest.raises(ValueError, match=r"positions \[1\]"):
        await embed_texts(model, ["ok", "  "])

    model.aembed_documents.assert_not_called()


@pytest.mark.parametrize(
    ("error", "permanent"),
    [
        (provider_error(400), True),  # e.g. "content contains an empty Part"
        (provider_error(401), True),
        (provider_error(403), True),
        (provider_error(404), True),
        (provider_error(408), False),
        (provider_error(429), False),  # rate limit / quota
        (provider_error(500), False),
        (provider_error(503), False),
        (ConnectionError("network down"), False),
        (TimeoutError(), False),
    ],
)
def test_only_provider_rejections_are_permanent(error, permanent):
    assert is_permanent_provider_error(error) is permanent
