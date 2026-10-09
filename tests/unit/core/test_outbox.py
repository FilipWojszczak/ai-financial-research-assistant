import pytest

from financial_assistant.core.outbox import publish_pending_documents

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("limit", [0, -1, True])
async def test_invalid_limit_is_rejected(limit):
    with pytest.raises(ValueError):
        await publish_pending_documents(limit=limit)
