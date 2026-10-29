import pgvector.psycopg
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from .. import models  # noqa: F401
from .config import get_settings

database_url = get_settings().database_url

engine = create_async_engine(database_url, echo=get_settings().database_echo_enabled)

async_session_maker = async_sessionmaker(
    engine,
    # Keeps objects in memory after commit. Required for FastAPI/Pydantic to read data
    # without triggering synchronous lazy loads.
    expire_on_commit=False,
)

_langgraph_pool: AsyncConnectionPool[AsyncConnection[DictRow]] | None = None


async def get_session():
    async with async_session_maker() as session:
        yield session


async def configure_pool_connection(conn):
    await pgvector.psycopg.register_vector_async(conn)


def libpq_url(sqlalchemy_url: str) -> str:
    """The SQLAlchemy URL in the plain libpq form that psycopg and LangGraph take."""
    return sqlalchemy_url.replace("postgresql+psycopg://", "postgresql://", 1)


def create_langgraph_pool(
    url: str,
) -> AsyncConnectionPool[AsyncConnection[DictRow]]:
    """An unopened pool in the shape AsyncPostgresSaver requires of a user pool."""
    return AsyncConnectionPool(
        conninfo=libpq_url(url),
        max_size=20,
        # The saver reads rows by column name (the default tuple rows raise
        # TypeError) and needs autocommit. Its statements are not prepared, so
        # they also work through transaction-mode poolers.
        kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": 0},
        connection_class=AsyncConnection[DictRow],
        open=False,
        configure=configure_pool_connection,
    )


async def init_langgraph_pool() -> AsyncConnectionPool[AsyncConnection[DictRow]]:
    global _langgraph_pool

    _langgraph_pool = create_langgraph_pool(database_url)
    await _langgraph_pool.open()
    return _langgraph_pool


async def close_langgraph_pool():
    global _langgraph_pool
    if _langgraph_pool:
        await _langgraph_pool.close()
        _langgraph_pool = None


def get_langgraph_pool() -> AsyncConnectionPool[AsyncConnection[DictRow]]:
    if _langgraph_pool is None:
        raise RuntimeError("LangGraph pool is not initialized. Check lifespan.")
    return _langgraph_pool
