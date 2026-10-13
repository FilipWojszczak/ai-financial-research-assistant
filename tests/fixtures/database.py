import os
import subprocess
import sys
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from financial_assistant.api.server import app
from financial_assistant.core.config import get_settings
from financial_assistant.core.db import engine as app_engine
from financial_assistant.core.db import get_session
from financial_assistant.core.security import create_access_token, hash_password
from financial_assistant.models import Base, Document, User
from financial_assistant.models.document import DocumentStatus, DocumentType
from tests.utils import DocumentFactory, TokenFactory, UserFactory

ROOT = Path(__file__).resolve().parents[2]
# Arbitrary fixed key; concurrent pytest sessions serialize on it in "postgres".
_MIGRATION_LOCK_KEY = 7_316_028_415


def _require_test_database(url: URL) -> str:
    """Return the test database name; exit unless every target ends in `_test`."""
    # Last line of defence against touching the dev database. core/db.py's engine is
    # checked too: it is built once at import time, so a redirect that ran too late
    # leaves it on the dev database while the settings already point at the test one.
    databases = {
        "the settings URL": url.database or "",
        "core/db.py's engine": app_engine.url.database or "",
    }
    for source, database in databases.items():
        if not database.endswith("_test"):
            pytest.exit(
                f"Refusing to run tests: {source} points at database {database!r}, "
                "whose name must end in '_test'. tests/conftest.py must set "
                "DATABASE_URL_OVERRIDE before core/db.py is imported.",
                returncode=pytest.ExitCode.USAGE_ERROR,
            )
    return databases["the settings URL"]


@pytest.fixture(scope="session")
def migrated_database() -> str:
    """Create `<dev db>_test` if missing and migrate it to head, once per session."""
    database_url = get_settings().database_url
    url = make_url(database_url)
    database = _require_test_database(url)
    # CREATE DATABASE can't run inside a transaction, hence AUTOCOMMIT.
    engine = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as connection:
            connection.execute(
                text("SELECT pg_advisory_lock(:key)"), {"key": _MIGRATION_LOCK_KEY}
            )
            try:
                exists = connection.execute(
                    text("SELECT 1 FROM pg_database WHERE datname = :name"),
                    {"name": database},
                ).scalar()
                if not exists:
                    name = engine.dialect.identifier_preparer.quote(database)
                    connection.execute(text(f"CREATE DATABASE {name}"))
                # A subprocess: alembic/env.py's fileConfig() would disable existing
                # loggers (breaking caplog), and sync Alembic stays out of the loop.
                result = subprocess.run(  # noqa: S603 - fixed argv, no shell
                    [sys.executable, "-m", "alembic", "upgrade", "head"],
                    check=False,
                    capture_output=True,
                    text=True,
                    cwd=ROOT,
                    # Migrate exactly the URL the guard checked.
                    env={**os.environ, "DATABASE_URL_OVERRIDE": database_url},
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"alembic upgrade head failed on {database!r}. If the "
                        "test database is at a revision missing from this branch, "
                        f'run DROP DATABASE "{database}" WITH (FORCE) and rerun '
                        f"pytest.\n{result.stderr}"
                    )
            finally:
                connection.execute(
                    text("SELECT pg_advisory_unlock(:key)"),
                    {"key": _MIGRATION_LOCK_KEY},
                )
    finally:
        engine.dispose()
    return database_url


@pytest_asyncio.fixture
async def private_database(migrated_database: str):
    # Subprocesses share this search path, isolated from existing application rows.
    schema = "ingestion_test_" + uuid.uuid4().hex
    url = (
        make_url(migrated_database)
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
async def session_fixture(migrated_database: str) -> AsyncGenerator[AsyncSession]:
    engine = create_async_engine(migrated_database)
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
