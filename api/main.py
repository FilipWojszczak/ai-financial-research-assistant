from contextlib import asynccontextmanager

from fastapi import FastAPI

from .db import close_langgraph_pool, init_langgraph_pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_langgraph_pool()
    yield
    await close_langgraph_pool()


app = FastAPI(
    title="AI Financial Research Assistant", version="0.1.0", lifespan=lifespan
)
