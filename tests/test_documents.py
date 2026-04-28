from unittest.mock import AsyncMock, patch

from httpx import AsyncClient
from tests.utils import DocumentFactory, TokenFactory, UserFactory

from financial_assistant.models.document import DocumentStatus

_PROCESS_TASK = "financial_assistant.api.routers.documents.process_uploaded_document"

_VALID_PDF = b"%PDF fake content"


# POST /documents/


async def test_upload_document_private(
    client: AsyncClient, user_factory: UserFactory, token_factory: TokenFactory
):
    user = await user_factory(email="uploader@example.com")
    token = token_factory(user)

    with patch(_PROCESS_TASK, new_callable=AsyncMock) as mock_process:
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
    mock_process.assert_called_once()


async def test_upload_document_public(
    client: AsyncClient, user_factory: UserFactory, token_factory: TokenFactory
):
    user = await user_factory(email="public_uploader@example.com")
    token = token_factory(user)

    with patch(_PROCESS_TASK, new_callable=AsyncMock):
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
    assert response.json()["owner_id"] is None


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


async def test_upload_document_no_filename(
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
    ids = {d["id"] for d in response.json()}
    assert own_doc.id in ids
    assert public_doc.id in ids
    assert other_doc.id not in ids


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

    response = await client.delete(
        f"/documents/{doc.id}", headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == 204


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
