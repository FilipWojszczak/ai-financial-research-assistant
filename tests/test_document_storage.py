from types import SimpleNamespace
from unittest.mock import patch

import pytest

from financial_assistant.core.document_storage import (
    delete_document_file,
    document_file_path,
    store_document_file,
)


def _settings(storage_path):
    return SimpleNamespace(document_storage_path=storage_path)


async def test_store_document_file_uses_stable_server_controlled_path(tmp_path):
    with patch(
        "financial_assistant.core.document_storage.get_settings",
        return_value=_settings(tmp_path),
    ):
        stored_path = await store_document_file(42, b"%PDF durable")

    assert stored_path == tmp_path / "42.pdf"
    assert stored_path.read_bytes() == b"%PDF durable"
    assert list(tmp_path.iterdir()) == [stored_path]


async def test_store_document_file_atomically_replaces_existing_source(tmp_path):
    with patch(
        "financial_assistant.core.document_storage.get_settings",
        return_value=_settings(tmp_path),
    ):
        await store_document_file(42, b"old")
        await store_document_file(42, b"new")

    assert (tmp_path / "42.pdf").read_bytes() == b"new"
    assert list(tmp_path.iterdir()) == [tmp_path / "42.pdf"]


async def test_delete_document_file_is_idempotent(tmp_path):
    source_path = tmp_path / "42.pdf"
    source_path.write_bytes(b"pdf")

    with patch(
        "financial_assistant.core.document_storage.get_settings",
        return_value=_settings(tmp_path),
    ):
        await delete_document_file(42)
        await delete_document_file(42)

    assert not source_path.exists()


@pytest.mark.parametrize("document_id", [0, -1])
def test_document_file_path_rejects_non_positive_ids(tmp_path, document_id):
    with (
        patch(
            "financial_assistant.core.document_storage.get_settings",
            return_value=_settings(tmp_path),
        ),
        pytest.raises(ValueError, match="positive integer"),
    ):
        document_file_path(document_id)
