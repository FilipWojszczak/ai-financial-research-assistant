"""
Tools the agent can call. To add one (e.g. web_search), append it to CHAT_TOOLS: it
returns sources through the same Command shape, and the grade and answer prompts
render any Source kind.
"""

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, Field, StringConstraints

from ...models.document import DocumentType
from .context import ChatContext
from .prompts import NO_DOCUMENTS_MATCH, render_search_results
from .state import ChatState


class RetrievalUnavailableError(Exception):
    """Search failed (database or embeddings); the turn fails, the LLM never sees it."""


class SearchDocumentsArgs(BaseModel):
    query: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=2, max_length=300)
    ] = Field(description="What to look for, as a self-contained search query")
    company_ticker: str | None = Field(
        default=None, description="Narrow the search to one company"
    )
    document_type: DocumentType | None = Field(
        default=None, description="Narrow the search to one document type"
    )
    year: int | None = Field(
        default=None, description="Narrow the search to one fiscal year"
    )


@tool("search_documents", args_schema=SearchDocumentsArgs)
async def search_documents(
    query: str,
    company_ticker: str | None = None,
    document_type: DocumentType | None = None,
    year: int | None = None,
    *,
    runtime: ToolRuntime[ChatContext, ChatState],
) -> Command:
    """Search the user's financial documents for passages and topic summaries."""
    context = runtime.context
    runtime.stream_writer({"type": "status", "stage": "searching", "query": query})
    # Filters from the LLM only ever narrow the scope built by the server.
    scope = context.scope.narrowed(
        company_ticker=company_ticker, document_type=document_type, year=year
    )
    if scope.is_empty:
        return _result(runtime, [], NO_DOCUMENTS_MATCH)
    try:
        # Its own session: ToolNode runs parallel tool calls concurrently.
        async with context.session_factory() as session:
            found = await context.retriever.search(session, query, scope)
    except Exception as exc:
        raise RetrievalUnavailableError from exc
    return _result(runtime, found, render_search_results(found))


def _result(
    runtime: ToolRuntime[ChatContext, ChatState], found: list, listing: str
) -> Command:
    message = ToolMessage(listing, tool_call_id=runtime.tool_call_id)
    return Command(update={"sources": found, "scratchpad": [message]})


CHAT_TOOLS: list[BaseTool] = [search_documents]
