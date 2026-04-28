from contextlib import asynccontextmanager

from fastapi import FastAPI

from ..core.db import close_langgraph_pool, init_langgraph_pool
from .routers import auth, documents


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_langgraph_pool()
    yield
    await close_langgraph_pool()


app = FastAPI(
    title="AI Financial Research Assistant", version="0.1.0", lifespan=lifespan
)

app.include_router(auth.router)
app.include_router(documents.router)
