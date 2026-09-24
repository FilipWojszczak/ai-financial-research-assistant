from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, UploadFile
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.datastructures import Headers
from tests.utils import DocumentFactory, TokenFactory, UserFactory

from financial_assistant.api.routers.documents import upload_document
from financial_assistant.models import DocumentOutbox
from financial_assistant.models.document import DocumentStatus, DocumentType
from financial_assistant.schemas.document import DocumentCreate

_PUBLISH_TASK = "financial_assistant.core.outbox.publish_ingestion"
_INGEST_DOCUMENT = "financial_assistant.ai.document_ingestion.ingest_document"
_STORE_FILE = "financial_assistant.api.routers.documents.store_document_file"
_DELETE_FILE = "financial_assistant.api.routers.documents.delete_document_file"

_VALID_PDF = b"%PDF fake content"


# POST /documents/


async def test_upload_document_private(
    client: AsyncClient,
    user_factory: UserFactory,
    token_factory: TokenFactory,
    session: AsyncSession,
):
    user = await user_factory(email="uploader@example.com")
    token = token_factory(user)

    with (
        patch(_PUBLISH_TASK, side_effect=ConnectionError("broker unavailable")) as send,
        patch(_INGEST_DOCUMENT, new_callable=AsyncMock) as ingest,
        patch(_STORE_FILE, new_callable=AsyncMock) as mock_store,
    ):
        response = await client.post(
            "/documents/",
            data={"company_ticker": "AAPL", "document_type": "10-K", "year": 2023},
            files={"file": ("annual_report.pdf", _VALID_PDF, "application/pdf")},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 202
    data = response.json()
    assert data["filename"] == "annual_report.pdf"
    assert data["company_ticker"] == "AAPL"
    assert data["document_type"] == "10-K"
    assert data["year"] == 2023
    assert data["status"] == DocumentStatus.PROCESSING
    assert data["owner_id"] == user.id
    mock_store.assert_awaited_once_with(data["id"], _VALID_PDF)
    send.assert_not_called()
    ingest.assert_not_awaited()
    event = await session.scalar(
        select(DocumentOutbox).where(DocumentOutbox.document_id == data["id"])
    )
    assert event is not None
    assert event.published_at is None
    assert event.attempts == 0


async def test_upload_document_public(
    client: AsyncClient,
    user_factory: UserFactory,
    token_factory: TokenFactory,
    session: AsyncSession,
):
    user = await user_factory(email="public_uploader@example.com")
    token = token_factory(user)

    with (
        patch(_PUBLISH_TASK, side_effect=ConnectionError("broker unavailable")) as send,
        patch(_INGEST_DOCUMENT, new_callable=AsyncMock) as ingest,
        patch(_STORE_FILE, new_callable=AsyncMock) as mock_store,
    ):
        response = await client.post(
            "/documents/",
            data={
                "company_ticker": "MSFT",
                "document_type": "10-Q",
                "is_public": "true",
            },
            files={"file": ("quarterly.pdf", _VALID_PDF, "application/pdf")},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 202
    data = response.json()
    assert data["owner_id"] is None
    mock_store.assert_awaited_once_with(data["id"], _VALID_PDF)
    send.assert_not_called()
    ingest.assert_not_awaited()
    event = await session.scalar(
        select(DocumentOutbox).where(DocumentOutbox.document_id == data["id"])
    )
    assert event is not None
    assert event.published_at is None
    assert event.attempts == 0


async def test_upload_document_wrong_content_type(
    client: AsyncClient, user_factory: UserFactory, token_factory: TokenFactory
):
    user = await user_factory(email="wrong_type@example.com")
    token = token_factory(user)

    response = await client.post(
        "/documents/",
        data={"company_ticker": "AAPL", "document_type": "10-K", "year": 2023},
        files={"file": ("report.txt", b"not a pdf", "text/plain")},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Only PDF files are allowed"


async def test_upload_document_empty_filename_fails_validation(
    client: AsyncClient, user_factory: UserFactory, token_factory: TokenFactory
):
    user = await user_factory(email="no_filename@example.com")
    token = token_factory(user)

    response = await client.post(
        "/documents/",
        data={"company_ticker": "AAPL", "document_type": "10-K", "year": 2023},
        files={"file": ("", _VALID_PDF, "application/pdf")},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "file"]
    assert error["type"] == "value_error"


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


async def test_upload_document_unauthenticated(client: AsyncClient):
    response = await client.post(
        "/documents/",
        data={"company_ticker": "AAPL", "document_type": "10-K", "year": 2023},
        files={"file": ("report.pdf", _VALID_PDF, "application/pdf")},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"


# GET /documents/


async def test_list_documents_returns_own_and_public_only(
    client: AsyncClient,
    user_factory: UserFactory,
    token_factory: TokenFactory,
    document_factory: DocumentFactory,
):
    user = await user_factory(email="list_owner@example.com")
    other_user = await user_factory(email="list_other@example.com")
    token = token_factory(user)

    own_doc = await document_factory(owner_id=user.id, company_ticker="AAPL")
    public_doc = await document_factory(owner_id=None, company_ticker="MSFT")
    other_doc = await document_factory(owner_id=other_user.id, company_ticker="GOOG")

    response = await client.get(
        "/documents/", headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == 200
    documents = response.json()
    ids = {document["id"] for document in documents}
    assert own_doc.id in ids
    assert public_doc.id in ids
    assert other_doc.id not in ids
    assert all(document["owner_id"] in {None, user.id} for document in documents)


async def test_list_documents_unauthenticated(client: AsyncClient):
    response = await client.get("/documents/")
    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"


# GET /documents/{document_id}


async def test_get_own_document(
    client: AsyncClient,
    user_factory: UserFactory,
    token_factory: TokenFactory,
    document_factory: DocumentFactory,
):
    user = await user_factory(email="get_own@example.com")
    token = token_factory(user)
    doc = await document_factory(owner_id=user.id)

    response = await client.get(
        f"/documents/{doc.id}", headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == 200
    assert response.json()["id"] == doc.id


async def test_get_public_document(
    client: AsyncClient,
    user_factory: UserFactory,
    token_factory: TokenFactory,
    document_factory: DocumentFactory,
):
    user = await user_factory(email="get_public@example.com")
    token = token_factory(user)
    public_doc = await document_factory(owner_id=None)

    response = await client.get(
        f"/documents/{public_doc.id}", headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == 200
    assert response.json()["id"] == public_doc.id


async def test_get_other_users_document_returns_404(
    client: AsyncClient,
    user_factory: UserFactory,
    token_factory: TokenFactory,
    document_factory: DocumentFactory,
):
    user = await user_factory(email="get_requester@example.com")
    other_user = await user_factory(email="get_doc_owner@example.com")
    token = token_factory(user)
    other_doc = await document_factory(owner_id=other_user.id)

    response = await client.get(
        f"/documents/{other_doc.id}", headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"


async def test_get_nonexistent_document_returns_404(
    client: AsyncClient, user_factory: UserFactory, token_factory: TokenFactory
):
    user = await user_factory(email="get_missing@example.com")
    token = token_factory(user)

    response = await client.get(
        "/documents/99999", headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"


async def test_get_document_unauthenticated(
    client: AsyncClient, user_factory: UserFactory, document_factory: DocumentFactory
):
    user = await user_factory(email="get_unauth_owner@example.com")
    doc = await document_factory(owner_id=user.id)

    response = await client.get(f"/documents/{doc.id}")
    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"


# DELETE /documents/{document_id}


async def test_delete_own_document(
    client: AsyncClient,
    user_factory: UserFactory,
    token_factory: TokenFactory,
    document_factory: DocumentFactory,
):
    user = await user_factory(email="delete_owner@example.com")
    token = token_factory(user)
    doc = await document_factory(owner_id=user.id)

    with patch(_DELETE_FILE, new_callable=AsyncMock) as mock_delete:
        response = await client.delete(
            f"/documents/{doc.id}", headers={"Authorization": f"Bearer {token}"}
        )

    assert response.status_code == 204
    mock_delete.assert_awaited_once_with(doc.id)

    get_response = await client.get(
        f"/documents/{doc.id}", headers={"Authorization": f"Bearer {token}"}
    )
    assert get_response.status_code == 404


async def test_delete_public_document_returns_403(
    client: AsyncClient,
    user_factory: UserFactory,
    token_factory: TokenFactory,
    document_factory: DocumentFactory,
):
    user = await user_factory(email="delete_public_requester@example.com")
    token = token_factory(user)
    public_doc = await document_factory(owner_id=None)

    response = await client.delete(
        f"/documents/{public_doc.id}", headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Public documents cannot be deleted"

    get_response = await client.get(
        f"/documents/{public_doc.id}", headers={"Authorization": f"Bearer {token}"}
    )
    assert get_response.status_code == 200


async def test_delete_other_users_document_returns_404(
    client: AsyncClient,
    user_factory: UserFactory,
    token_factory: TokenFactory,
    document_factory: DocumentFactory,
):
    user = await user_factory(email="delete_requester@example.com")
    other_user = await user_factory(email="delete_doc_owner@example.com")
    token = token_factory(user)
    other_doc = await document_factory(owner_id=other_user.id)

    response = await client.delete(
        f"/documents/{other_doc.id}", headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"

    other_token = token_factory(other_user)
    get_response = await client.get(
        f"/documents/{other_doc.id}",
        headers={"Authorization": f"Bearer {other_token}"},
    )
    assert get_response.status_code == 200


async def test_delete_nonexistent_document_returns_404(
    client: AsyncClient, user_factory: UserFactory, token_factory: TokenFactory
):
    user = await user_factory(email="delete_missing@example.com")
    token = token_factory(user)

    response = await client.delete(
        "/documents/99999", headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"


async def test_delete_document_unauthenticated(
    client: AsyncClient, user_factory: UserFactory, document_factory: DocumentFactory
):
    user = await user_factory(email="delete_unauth_owner@example.com")
    doc = await document_factory(owner_id=user.id)

    response = await client.delete(f"/documents/{doc.id}")
    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"
