import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from financial_assistant.ai.requests import ai_request, embed_texts


async def test_ai_deadline_cancels_the_pending_call():
    cancelled = asyncio.Event()

    async def hang():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    with (
        patch(
            "financial_assistant.ai.requests.get_settings",
            return_value=SimpleNamespace(ai_request_timeout_seconds=0.01),
        ),
        pytest.raises(TimeoutError),
    ):
        await ai_request(hang())
    assert cancelled.is_set()


async def test_embedding_deadline_applies_to_each_batch_not_total_work():
    async def embed(batch):
        await asyncio.sleep(0.04)
        return [[float(value)] for value in batch]

    model = SimpleNamespace(aembed_documents=AsyncMock(side_effect=embed))
    texts = [str(i) for i in range(65)]
    with patch(
        "financial_assistant.ai.requests.get_settings",
        return_value=SimpleNamespace(ai_request_timeout_seconds=0.15),
    ):
        result = await embed_texts(model, texts)
    assert result == [[float(i)] for i in range(65)]
    assert model.aembed_documents.await_count == 5
    assert all(
        len(call.args[0]) <= 16 for call in model.aembed_documents.await_args_list
    )
