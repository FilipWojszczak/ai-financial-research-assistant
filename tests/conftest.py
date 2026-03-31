from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from api.config import get_settings
from api.db import get_session
from api.main import app
from api.models import User
from api.utils import create_access_token, hash_password
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from tests.utils import TokenFactory, UserFactory


@pytest_asyncio.fixture(name="session")
async def session_fixture() -> AsyncGenerator[AsyncSession]:
    database_url = get_settings().database_url
    if "postgres" in database_url:
        engine = create_async_engine(database_url)
    else:
        raise ValueError("database_url must be set to a PostgreSQL database for tests.")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    connection = await engine.connect()
    transaction = await connection.begin()

    session = AsyncSession(bind=connection)

    yield session

    await session.close()
    await transaction.rollback()
    await connection.close()
    await engine.dispose()


@pytest_asyncio.fixture(name="client")
async def client_fixture(session: AsyncSession) -> AsyncGenerator[AsyncClient]:
    async def get_session_override():
        return session

    app.dependency_overrides[get_session] = get_session_override

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", follow_redirects=True
    ) as client:
        yield client

    app.dependency_overrides.clear()


@pytest_asyncio.fixture(name="user_factory")
async def user_factory_fixture(
    session: AsyncSession,
) -> UserFactory:
    async def _create_user(email: str, password: str = "securepassword") -> User:  # noqa: S107
        hashed_password = hash_password(password)
        user = User(email=email, hashed_password=hashed_password)
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user

    return _create_user


@pytest.fixture(name="token_factory")
def token_factory_fixture() -> TokenFactory:
    def _create_token(user: User) -> str:
        return create_access_token(user.id)

    return _create_token
