import os
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from financial_assistant import cleanup_storage

pytestmark = pytest.mark.unit


def old_file(path):
    path.write_bytes(b"source")
    past = time.time() - 3600
    os.utime(path, (past, past))
    return path


async def test_database_failure_never_authorizes_file_deletion(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cleanup_storage,
        "get_settings",
        lambda: SimpleNamespace(document_storage_path=tmp_path),
    )
    source = old_file(tmp_path / "999.pdf")
    failed_session = MagicMock()
    failed_session.__aenter__ = AsyncMock(side_effect=ConnectionError("database down"))
    with (
        patch.object(
            cleanup_storage, "async_session_maker", return_value=failed_session
        ),
        pytest.raises(ConnectionError),
    ):
        await cleanup_storage.cleanup_orphaned_files(min_age_seconds=60)
    assert source.exists()
