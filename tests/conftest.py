import uuid
from collections.abc import AsyncGenerator
from types import SimpleNamespace

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from tests.utils import DocumentFactory, TokenFactory, UserFactory

from financial_assistant.api.server import app
from financial_assistant.core.config import get_settings
from financial_assistant.core.db import get_session
from financial_assistant.models import Base, Document, User
from financial_assistant.models.document import DocumentStatus, DocumentType
from financial_assistant.utils import create_access_token, hash_password


@pytest_asyncio.fixture
async def private_database():
    # Subprocesses share this search path, isolated from existing application rows.
    schema = "ingestion_test_" + uuid.uuid4().hex
    url = (
        make_url(get_settings().database_url)
        .update_query_dict({"options": f"-csearch_path={schema},public"})
        .render_as_string(hide_password=False)
    )
    engine = create_async_engine(
        url, execution_options={"schema_translate_map": {None: schema}}
    )
    try:
        async with engine.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            await connection.run_sync(Base.metadata.create_all)
        yield SimpleNamespace(
            url=SecretStr(url),
            factory=async_sessionmaker(engine, expire_on_commit=False),
        )
    finally:
        async with engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await engine.dispose()


@pytest_asyncio.fixture(name="session")
async def session_fixture() -> AsyncGenerator[AsyncSession]:
    database_url = get_settings().database_url
    if "postgres" in database_url:
        engine = create_async_engine(database_url)
    else:
        raise ValueError("database_url must be set to a PostgreSQL database for tests.")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    connection = await engine.connect()
    transaction = await connection.begin()

    session = AsyncSession(bind=connection, expire_on_commit=False)

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
    async def _create_user(email: str, password: str = "securepassword") -> User:
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


@pytest_asyncio.fixture(name="document_factory")
async def document_factory_fixture(session: AsyncSession) -> DocumentFactory:
    async def _create(
        owner_id: int | None = None,
        company_ticker: str = "AAPL",
        document_type: DocumentType = DocumentType.ANNUAL_REPORT,
        year: int = 2023,
        filename: str = "test.pdf",
        status: DocumentStatus = DocumentStatus.COMPLETED,
    ) -> Document:
        document = Document(
            filename=filename,
            company_ticker=company_ticker,
            document_type=document_type,
            year=year,
            owner_id=owner_id,
            status=status,
        )
        session.add(document)
        await session.commit()
        await session.refresh(document)
        return document

    return _create
