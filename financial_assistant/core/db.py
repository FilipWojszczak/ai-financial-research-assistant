import pgvector.psycopg
from psycopg_pool import AsyncConnectionPool
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from .. import models  # noqa: F401
from .config import get_settings

database_url = get_settings().database_url

engine = create_async_engine(database_url, echo=True)

_langgraph_pool: AsyncConnectionPool | None = None


async def get_session():
    async with AsyncSession(engine) as session:
        yield session


async def configure_pool_connection(conn):
    await pgvector.psycopg.register_vector_async(conn)


async def init_langgraph_pool() -> AsyncConnectionPool:
    global _langgraph_pool

    db_url = database_url.replace("postgresql+psycopg://", "postgresql://")

    _langgraph_pool = AsyncConnectionPool(
        conninfo=db_url,
        max_size=20,
        kwargs={"autocommit": True},
        open=False,
        configure=configure_pool_connection,
    )
    await _langgraph_pool.open()
    return _langgraph_pool


async def close_langgraph_pool():
    global _langgraph_pool
    if _langgraph_pool:
        await _langgraph_pool.close()
        _langgraph_pool = None


def get_langgraph_pool() -> AsyncConnectionPool:
    if _langgraph_pool is None:
        raise RuntimeError("LangGraph pool is not initialized. Check lifespan.")
    return _langgraph_pool
