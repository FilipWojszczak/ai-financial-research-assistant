from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers

from financial_assistant.api.routers.documents import upload_document
from financial_assistant.models.document import DocumentType
from financial_assistant.schemas.document import DocumentCreate

pytestmark = pytest.mark.unit

_STORE_FILE = "financial_assistant.api.routers.documents.store_document_file"
_DELETE_FILE = "financial_assistant.api.routers.documents.delete_document_file"

_VALID_PDF = b"%PDF fake content"


async def test_upload_document_rejects_upload_file_without_filename():
    """The endpoint's defensive filename check returns its custom validation error."""
    file = UploadFile(
        file=BytesIO(_VALID_PDF),
        filename="",
        headers=Headers({"content-type": "application/pdf"}),
    )

    with pytest.raises(HTTPException) as exc_info:
        await upload_document(
            document_data=DocumentCreate(
                company_ticker="AAPL",
                document_type=DocumentType.ANNUAL_REPORT,
                year=2023,
            ),
            file=file,
            session=AsyncMock(),
            user=MagicMock(id=1),
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "File must have a filename"


@pytest.mark.parametrize("failure_stage", ["flush", "commit"])
async def test_upload_failure_cleans_up_only_before_commit(failure_stage):
    """Keep source data when the outcome of COMMIT cannot be known."""
    file = UploadFile(
        file=BytesIO(_VALID_PDF),
        filename="report.pdf",
        headers=Headers({"content-type": "application/pdf"}),
    )
    session = MagicMock()
    session.scalar = AsyncMock()

    async def assign_document_id():
        if session.flush.await_count == 1:
            session.add.call_args_list[0].args[0].id = 42
        elif failure_stage == "flush":
            raise RuntimeError("commit failed")

    session.flush = AsyncMock(side_effect=assign_document_id)
    session.commit = AsyncMock(side_effect=RuntimeError("commit failed"))
    session.rollback = AsyncMock()
    session.refresh = AsyncMock()

    with (
        patch(_STORE_FILE, new_callable=AsyncMock) as mock_store,
        patch(_DELETE_FILE, new_callable=AsyncMock) as mock_delete,
        pytest.raises(RuntimeError, match="commit failed"),
    ):
        await upload_document(
            document_data=DocumentCreate(
                company_ticker="AAPL",
                document_type=DocumentType.ANNUAL_REPORT,
                year=2023,
            ),
            file=file,
            session=session,
            user=MagicMock(id=1),
        )

    assert session.flush.await_count == 2
    mock_store.assert_awaited_once_with(42, _VALID_PDF)
    session.rollback.assert_awaited_once_with()
    if failure_stage == "flush":
        mock_delete.assert_awaited_once_with(42)
        session.commit.assert_not_awaited()
    else:
        mock_delete.assert_not_awaited()
    session.refresh.assert_not_awaited()
