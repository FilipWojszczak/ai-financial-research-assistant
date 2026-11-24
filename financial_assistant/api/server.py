from contextlib import asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from ..agents.chat.graph import build_chat_graph
from ..agents.chat.models import create_chat_models
from ..agents.chat.runner import ChatTurnRunner, cancel_running_turns
from ..agents.chat.state import chat_serializer
from ..core.config import get_settings
from ..core.db import async_session_maker, close_langgraph_pool, init_langgraph_pool
from ..core.tracing import shutdown_tracing
from ..retrieval.hybrid import default_retriever
from .routers import auth, chat, documents


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    pool = await init_langgraph_pool()
    try:
        # Its tables are created by services/setup_checkpointer, not here.
        checkpointer = AsyncPostgresSaver(conn=pool, serde=chat_serializer())
        graph = build_chat_graph(
            create_chat_models(settings), checkpointer=checkpointer
        )
        app.state.checkpointer = checkpointer
        app.state.chat_runner = ChatTurnRunner(
            graph, async_session_maker, default_retriever(settings), settings
        )
        yield
    finally:
        # Turns first: they still need the pool and the tracing client to finish.
        await cancel_running_turns()
        shutdown_tracing()
        await close_langgraph_pool()


app = FastAPI(
    title="AI Financial Research Assistant", version="0.1.0", lifespan=lifespan
)

app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)
