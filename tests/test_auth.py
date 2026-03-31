from httpx import AsyncClient
from tests.utils import TokenFactory, UserFactory


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
