import logging
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_assistant.ai.document_ingestion import INGESTION_LOCK
from financial_assistant.api.server import app
from financial_assistant.core import document_storage
from financial_assistant.core.accounts import TOMBSTONE_PASSWORD_HASH, tombstone_email
from financial_assistant.core.db import get_session
from financial_assistant.core.security import hash_password
from financial_assistant.models import Document, ParentChunk, User
from financial_assistant.models.document import DocumentType
from tests.utils import DocumentFactory, TokenFactory, UserFactory

pytestmark = pytest.mark.integration


async def test_authentication(client: AsyncClient):
    email = "john.smith@example.com"
    password = "securepassword"

    # Register a new user
    response = await client.post(
        "/auth/register",
        json={"email": email, "password": password},
    )
    data = response.json()
    assert response.status_code == 201
    assert data["email"] == email
    assert data["is_active"] is True
    assert isinstance(data["id"], int)

    # Log in to obtain an access token
    response = await client.post(
        "/auth/token",
        data={"username": email, "password": password},
    )
    data = response.json()
    assert response.status_code == 200
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    access_token = data["access_token"]

    # Access a protected endpoint
    response = await client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    data = response.json()
    assert response.status_code == 200
    assert data["email"] == email
    assert data["is_active"] is True
    assert isinstance(data["id"], int)


async def test_register_existing_user(client: AsyncClient):
    email = "john.smith@example.com"
    password = "securepassword"

    response = await client.post(
        "/auth/register",
        json={"email": email, "password": password},
    )
    assert response.status_code == 201

    # Attempt to register the same user again
    response = await client.post(
        "/auth/register",
        json={"email": email, "password": password},
    )
    data = response.json()
    assert response.status_code == 409
    assert data["detail"] == "User with this email already exists"


async def test_login_invalid_credentials(client: AsyncClient):
    email = "john.smith@example.com"
    password = "securepassword"
    not_existing_email = "abc@example.com"
    wrong_password = "wrongpassword"

    # Register a new user
    response = await client.post(
        "/auth/register",
        json={"email": email, "password": password},
    )
    assert response.status_code == 201

    # Log in with incorrect password
    response = await client.post(
        "/auth/token",
        data={"username": email, "password": wrong_password},
    )
    data = response.json()
    assert response.status_code == 401
    assert "access_token" not in data
    assert data["detail"] == "Incorrect email or password"

    # Log in with non-existing email
    response = await client.post(
        "/auth/token",
        data={"username": not_existing_email, "password": password},
    )
    data = response.json()
    assert response.status_code == 401
    assert "access_token" not in data
    assert data["detail"] == "Incorrect email or password"


async def test_access_protected_endpoint_no_token(client: AsyncClient):
    # Attempt to access a protected endpoint without a token
    response = await client.get("/auth/me")
    data = response.json()
    assert response.status_code == 401
    assert data["detail"] == "Not authenticated"


async def test_update_password_success(
    client: AsyncClient, user_factory: UserFactory, token_factory: TokenFactory
):
    # Create a user and obtain a token
    current_password = "securepassword123"
    new_password = "newsecurepassword123"
    user = await user_factory(
        email="update_pass@example.com", password=current_password
    )
    access_token = token_factory(user)

    # Update the password
    response = await client.put(
        "/auth/me/password",
        json={"current_password": current_password, "new_password": new_password},
        headers={"Authorization": f"Bearer {access_token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "update_pass@example.com"

    # Log in with the new password
    login_response = await client.post(
        "/auth/token",
        data={"username": "update_pass@example.com", "password": new_password},
    )
    assert login_response.status_code == 200
    assert "access_token" in login_response.json()

    old_password_response = await client.post(
        "/auth/token",
        data={"username": "update_pass@example.com", "password": current_password},
    )
    assert old_password_response.status_code == 401


async def test_update_password_incorrect_current_password(
    client: AsyncClient, user_factory: UserFactory, token_factory: TokenFactory
):
    # Create a user and obtain a token
    current_password = "securepassword123"
    user = await user_factory(email="wrong_pass@example.com", password=current_password)
    access_token = token_factory(user)

    # Attempt to update the password with incorrect current password
    response = await client.put(
        "/auth/me/password",
        json={
            "current_password": "wrongpassword123",
            "new_password": "newsecurepassword123",
        },
        headers={"Authorization": f"Bearer {access_token}"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Current password is incorrect"

    unchanged_login = await client.post(
        "/auth/token",
        data={"username": "wrong_pass@example.com", "password": current_password},
    )
    assert unchanged_login.status_code == 200


async def test_update_password_same_as_current(
    client: AsyncClient, user_factory: UserFactory, token_factory: TokenFactory
):
    # Create a user and obtain a token
    current_password = "securepassword123"
    user = await user_factory(email="same_pass@example.com", password=current_password)
    access_token = token_factory(user)

    # Attempt to update the password with the same password
    response = await client.put(
        "/auth/me/password",
        json={"current_password": current_password, "new_password": current_password},
        headers={"Authorization": f"Bearer {access_token}"},
    )

    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "Current password and new password cannot be the same"
    )

    unchanged_login = await client.post(
        "/auth/token",
        data={"username": "same_pass@example.com", "password": current_password},
    )
    assert unchanged_login.status_code == 200


async def test_delete_account_success(
    client: AsyncClient, user_factory: UserFactory, token_factory: TokenFactory
):
    # Create a user and obtain a token
    password = "securepassword123"
    email = "delete_me@example.com"
    user = await user_factory(email=email, password=password)
    access_token = token_factory(user)

    # Delete the user account
    response = await client.request(
        "DELETE",
        "/auth/me",
        json={"password": password},
        headers={"Authorization": f"Bearer {access_token}"},
    )

    assert response.status_code == 204

    # Attempt to log in with the deleted account
    login_response = await client.post(
        "/auth/token",
        data={"username": email, "password": password},
    )
    assert login_response.status_code == 401


async def test_delete_account_incorrect_password(
    client: AsyncClient, user_factory: UserFactory, token_factory: TokenFactory
):
    password = "securepassword123"
    user = await user_factory(email="keep_me@example.com", password=password)
    access_token = token_factory(user)

    # Attempt to delete the account with incorrect password
    response = await client.request(
        "DELETE",
        "/auth/me",
        json={"password": "wrongpassword123"},
        headers={"Authorization": f"Bearer {access_token}"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect password"

    unchanged_login = await client.post(
        "/auth/token",
        data={"username": "keep_me@example.com", "password": password},
    )
    assert unchanged_login.status_code == 200


_ROUTER_DELETE_FILE = "financial_assistant.api.routers.auth.delete_document_file"


async def _delete_me(client: AsyncClient, token: str, **body: object):
    return await client.request(
        "DELETE",
        "/auth/me",
        json={"password": "securepassword", **body},
        headers={"Authorization": f"Bearer {token}"},
    )


async def _user_row(session: AsyncSession, user_id: int):
    result = await session.execute(
        select(User.email, User.hashed_password, User.is_active, User.deleted_at).where(
            User.id == user_id
        )
    )
    return result.one_or_none()


async def test_delete_account_without_documents_removes_the_user(
    client: AsyncClient,
    session: AsyncSession,
    user_factory: UserFactory,
    token_factory: TokenFactory,
):
    user = await user_factory(email="no_docs@example.com")

    response = await _delete_me(client, token_factory(user))

    assert response.status_code == 204
    assert await _user_row(session, user.id) is None


async def test_delete_documents_without_documents_removes_the_user(
    client: AsyncClient,
    session: AsyncSession,
    user_factory: UserFactory,
    token_factory: TokenFactory,
):
    user = await user_factory(email="no_docs_delete_all@example.com")

    with patch(_ROUTER_DELETE_FILE, new_callable=AsyncMock) as remove:
        response = await _delete_me(client, token_factory(user), delete_documents=True)

    assert response.status_code == 204
    assert await _user_row(session, user.id) is None
    remove.assert_not_awaited()


async def test_delete_account_with_documents_tombstones_the_user(
    client: AsyncClient,
    session: AsyncSession,
    user_factory: UserFactory,
    token_factory: TokenFactory,
    document_factory: DocumentFactory,
):
    email = "has_docs@example.com"
    owner = await user_factory(email=email)
    other = await user_factory(email="other@example.com")
    other_token = token_factory(other)
    owner_token = token_factory(owner)
    private = await document_factory(owner_id=owner.id, company_ticker="PRIV")
    public = await document_factory(owner_id=None, company_ticker="PUBL")

    response = await _delete_me(client, owner_token)

    assert response.status_code == 204
    row = await _user_row(session, owner.id)
    assert row is not None
    assert row.email == tombstone_email(owner.id)
    assert row.hashed_password == TOMBSTONE_PASSWORD_HASH
    assert row.is_active is False
    assert row.deleted_at is not None
    # Regression: the documents keep their owner and never become public.
    owners = await session.scalars(
        select(Document.owner_id).where(Document.id.in_([private.id, public.id]))
    )
    assert sorted(owners, key=lambda owner_id: owner_id is None) == [owner.id, None]

    headers = {"Authorization": f"Bearer {other_token}"}
    listed = await client.get("/documents/", headers=headers)
    assert [doc["id"] for doc in listed.json()] == [public.id]
    assert (
        await client.get(f"/documents/{private.id}", headers=headers)
    ).status_code == 404
    assert (
        await client.get(f"/documents/{public.id}", headers=headers)
    ).status_code == 200

    # The old token and the old credentials no longer work.
    me = await client.get(
        "/auth/me", headers={"Authorization": f"Bearer {owner_token}"}
    )
    assert me.status_code == 403
    login = await client.post(
        "/auth/token", data={"username": email, "password": "securepassword"}
    )
    assert login.status_code == 401

    # The email is free again; the new account cannot see the old documents.
    registered = await client.post(
        "/auth/register", json={"email": email, "password": "securepassword"}
    )
    assert registered.status_code == 201
    assert registered.json()["id"] != owner.id
    login = await client.post(
        "/auth/token", data={"username": email, "password": "securepassword"}
    )
    new_headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    listed = await client.get("/documents/", headers=new_headers)
    assert [doc["id"] for doc in listed.json()] == [public.id]


async def test_delete_account_with_delete_documents_removes_everything(
    client: AsyncClient,
    session: AsyncSession,
    user_factory: UserFactory,
    token_factory: TokenFactory,
    document_factory: DocumentFactory,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        document_storage,
        "get_settings",
        lambda: SimpleNamespace(document_storage_path=tmp_path),
    )
    owner = await user_factory(email="delete_all@example.com")
    documents = [await document_factory(owner_id=owner.id) for _ in range(2)]
    public = await document_factory(owner_id=None)
    for document in [*documents, public]:
        (tmp_path / f"{document.id}.pdf").write_bytes(b"%PDF")
    session.add(ParentChunk(chunk_index=0, content="text", document_id=documents[0].id))
    await session.commit()

    response = await _delete_me(client, token_factory(owner), delete_documents=True)

    assert response.status_code == 204
    assert await _user_row(session, owner.id) is None
    ids = [document.id for document in documents]
    assert (
        await session.scalar(
            select(func.count()).select_from(Document).where(Document.id.in_(ids))
        )
        == 0
    )
    assert (
        await session.scalar(
            select(func.count())
            .select_from(ParentChunk)
            .where(ParentChunk.document_id.in_(ids))
        )
        == 0
    )
    assert not any((tmp_path / f"{document_id}.pdf").exists() for document_id in ids)
    assert (tmp_path / f"{public.id}.pdf").exists()
    assert (
        await session.scalar(select(Document.id).where(Document.id == public.id))
        == public.id
    )


async def test_delete_account_succeeds_when_pdf_removal_fails(
    client: AsyncClient,
    session: AsyncSession,
    user_factory: UserFactory,
    token_factory: TokenFactory,
    document_factory: DocumentFactory,
    caplog: pytest.LogCaptureFixture,
):
    owner = await user_factory(email="pdf_fails@example.com")
    document = await document_factory(owner_id=owner.id)

    with (
        patch(_ROUTER_DELETE_FILE, new_callable=AsyncMock) as remove,
        caplog.at_level(logging.WARNING),
    ):
        remove.side_effect = OSError("disk gone")
        response = await _delete_me(client, token_factory(owner), delete_documents=True)

    assert response.status_code == 204
    remove.assert_awaited_once_with(document.id)
    assert await _user_row(session, owner.id) is None
    assert f"deleted document {document.id} (OSError)" in caplog.text


async def test_delete_documents_while_one_is_ingesting_returns_409(
    client: AsyncClient, private_database, token_factory: TokenFactory
):
    # Needs committed rows and a second connection holding the worker's row lock.
    factory = private_database.factory

    async def private_session():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_session] = private_session
    async with factory() as setup:
        owner = User(
            email="busy@example.com", hashed_password=hash_password("securepassword")
        )
        setup.add(owner)
        await setup.flush()
        documents = [
            Document(
                filename=f"{index}.pdf",
                company_ticker="BUSY",
                document_type=DocumentType.ANNUAL_REPORT,
                year=2026,
                owner_id=owner.id,
            )
            for index in range(2)
        ]
        setup.add_all(documents)
        await setup.commit()
    ids = sorted(document.id for document in documents)

    async with factory() as worker, worker.begin():
        await worker.get(Document, ids[0], with_for_update=INGESTION_LOCK)
        started = time.monotonic()
        response = await _delete_me(client, token_factory(owner), delete_documents=True)
        elapsed = time.monotonic() - started

    assert response.status_code == 409
    assert response.json()["detail"].startswith("A document is still being processed")
    # The delete waited for the lock up to DOCUMENT_LOCK_TIMEOUT (5s) instead of failing
    # fast (> 4s), and did not hang (< 15s, slack for slow CI).
    assert 4 < elapsed < 15
    async with factory() as check:
        user = await check.get(User, owner.id)
        assert user is not None
        assert (user.email, user.is_active, user.deleted_at) == (
            "busy@example.com",
            True,
            None,
        )
        owners = await check.scalars(
            select(Document.owner_id).where(Document.id.in_(ids))
        )
        assert list(owners) == [owner.id, owner.id]
