# Hybrid RAG chat (LangGraph agent + Langfuse tracing)

## Goal

Authenticated users chat with the financial documents they can access (public + own).
Each turn runs a LangGraph agent:

1. it searches the documents with **hybrid retrieval**: ParadeDB BM25 and pgvector dense
   search over `child_chunk`, fused with Reciprocal Rank Fusion and expanded to
   `parent_chunk` context, plus dense search over the GraphRAG `graph_community`
   summaries;
2. a grader checks whether the retrieved context is sufficient (at most 2 retrieval
   rounds);
3. a separate node streams a cited answer over Server-Sent Events.

Conversations persist across turns (LangGraph Postgres checkpointer). Every chat turn
**and** every ingestion attempt is traced in Langfuse Cloud. A per-user daily token
budget bounds spend.

Decisions taken during the interview:

| Topic | Decision |
| --- | --- |
| Graph shape | Tool-calling agent + grader: `agent` (LLM with bound tools) → `tools` → `grade` → back to `agent` (max 2 retrieval rounds) → separate streamed `generate` node. Tools are a registry, so web search can be added later as one more tool |
| First round | Forced search: `tool_choice="any"` on round 1; round 2 is `auto` |
| Retrieval sources | Child chunks (BM25 + dense, RRF) → parent chunks; community summaries (dense). A `Retriever` protocol lets an entity/local-graph retriever plug in later |
| Fusion | RRF, k=60, equal weights; top 20 per list → top 8 children → ≤5 parents; communities: dense top 3 above a similarity floor. All numbers are settings |
| Document scope | Optional thread filters (`document_ids`, `company_ticker`) fixed at thread creation; access control always enforced server-side; the LLM can only narrow, never widen |
| Response | SSE via FastAPI's native `EventSourceResponse` (FastAPI 0.135.2) |
| Persistence | `AsyncPostgresSaver` for graph state + Alembic-managed `chat_thread` and `chat_turn` tables |
| Langfuse | Langfuse Cloud, SDK v4, full input/output capture, disabled when keys are absent |
| Tracing scope | Chat turns and Celery ingestion (one trace per attempt, every LLM call and embedding batch) |
| Models | `gemini-3.5-flash-lite` for agent and grader, `gemini-3.6-flash` for the answer; names are settings |
| Rate limit | Daily per-user token budget, summed from `chat_turn` |
| Thread API | Create, list, get with history, delete, send message (SSE) |
| Vector index | HNSW (`vector_cosine_ops`) on `child_chunk.embedding` |
| Eval | Synthetic fictional-company 10-K PDF + golden Q&A set, run as an `llm` test |
| SQL logging | New `database_echo` setting; `core/db.py` stops hard-coding `echo=True` |
| Account deletion | Handled by the prerequisite fix; chat data is deleted by the shared `delete_account` helper |

**Prerequisite:** implement
[orphaned-documents-on-account-deletion](orphaned-documents-on-account-deletion.md)
first, on its own `fix/` branch. Without it, deleting an account makes that user's private
documents public, and this feature would make them searchable by everyone. After it,
a deleted user's documents keep their `owner_id` and are unreachable by construction.

Decided without asking (pushback welcome): a second concurrent message on the same
thread gets `409`; a client disconnect cancels the run and the turn is not saved;
history keeps only question/answer pairs (tool calls and results live in a per-turn
scratchpad); `user_id` and thread filters are never tool arguments visible to the LLM.

## Non-goals

- **Web search tool.** Only the extension point (tool registry, `Source` union) is
  built.
- **Entity / local-graph retriever.** Entity embeddings are never populated today
  (0 of 3113 `entity` rows in the dev DB have one). A later spec must backfill them
  (~3k embedding calls for the current 10 documents) or match entities by name.
- LLM reranking, LLM-extracted metadata filters, HyDE or multi-query expansion.
- Any change to chunking (parent 2000/200, child 400/50), the embedding model or its
  768 dimensions, entity dedup or community detection. **No re-embedding.**
- Vector indexes on `graph_community.embedding` (312 rows) or `entity.embedding`.
- A frontend. Editing/regenerating messages, branching or time travel, human-in-the-loop
  interrupts.
- Conversation summarization memory: only a sliding history window.
- Turn-count rate limits, per-IP limits, a global spend cap.
- Langfuse self-hosting, Langfuse prompt management, Langfuse datasets or LLM-as-judge
  experiments, input/output masking.
- Changing a thread's filters after creation; sharing threads between users.
- Fixing the account-deletion behaviour for documents: that is the prerequisite spec.

## Design

### Overview

```
POST /chat/threads/{id}/messages
  │  auth · thread ownership · thread-length cap · daily budget · claim chat_turn (409 if busy)
  ▼
ChatTurnRunner  (own asyncio task → queue → SSE generator)
  │  Langfuse trace: session = thread id, user = user id, tags ["chat"]
  ▼
START → prepare → agent ──tool_calls──► tools ──► grade ──insufficient & round < 2──► agent
                    │                               │
                    └── no tool_calls ──┐           └── sufficient or cap ──┐
                                        ▼                                   ▼
                                     generate (streams answer tokens) ───► END
```

New modules (the planned `financial_assistant/retrieval/**` and
`financial_assistant/agents/**` paths in `.claude/rules/llm-pipeline.md`):

```
financial_assistant/
  ai/embeddings.py            # single source of truth for the embedding model
  core/tracing.py             # Langfuse client, callbacks, trace scopes (no-op when disabled)
  retrieval/{__init__,types,scope,fusion,chunks,communities,hybrid}.py
  agents/__init__.py
  agents/chat/{__init__,state,context,models,prompts,tools,nodes,graph,runner,turns}.py
  models/chat.py              # ChatThread, ChatTurn, ChatTurnStatus
  schemas/chat.py             # request/response + SSE payload models
  api/routers/chat.py
  api/dependencies/chat.py
  services/setup_checkpointer.py
```

Changed modules: `core/config.py`, `core/db.py`, `api/server.py`,
`core/accounts.py` (extended from the prerequisite fix to delete chat data), `ai/requests.py`,
`ai/document_ingestion.py`, `ai/community_detection.py`, `ai/graph_extraction.py`,
`ai/checkpoints.py`, `tasks/document_ingestion.py`, `tasks/async_runner.py`,
`models/document.py` (HNSW index), `models/__init__.py`, `docker-compose.yml`
(`migrate` command), `.env.example`, `CLAUDE.md`, `.claude/rules/llm-pipeline.md`,
`.claude/rules/database.md`, `tests/conftest.py`, `tests/fixtures/database.py`.

This touches far more than 2–3 files. Implement it in the phases listed in the
[Test plan](#test-plan), each with green checks before the next.

### New dependencies

```
uv add langgraph-checkpoint-postgres langfuse
```

As of 2026-10-14 on PyPI: `langgraph-checkpoint-postgres` 3.1.3, which requires
`langgraph-checkpoint>=4.3.0`. Version 4.0.1 is installed, so uv upgrades it. `langfuse`
is 4.18.0 (SDK v4, OpenTelemetry-based). Do this as its own first step: record
`uv pip show langgraph langgraph-checkpoint langgraph-checkpoint-postgres langfuse`,
then run the fast suite before writing feature code, so breakage from the checkpoint
upgrade is isolated. `reportlab` is **not** added; the golden-PDF generator runs with
`uv run --with reportlab`.

### Settings (`core/config.py`, `.env.example`)

All new settings go into `.env.example` (with their defaults), in the same flat style as
the existing ones.

| Setting | Default | Notes |
| --- | --- | --- |
| `langfuse_public_key` | `None` | Tracing is enabled only when both keys are set |
| `langfuse_secret_key` | `None` | `SecretStr` |
| `langfuse_base_url` | `https://cloud.langfuse.com` | EU region; US is `https://us.cloud.langfuse.com` |
| `chat_agent_model` | `gemini-3.5-flash-lite` | temperature 0, lowest thinking level |
| `chat_grader_model` | `gemini-3.5-flash-lite` | temperature 0, lowest thinking level |
| `chat_answer_model` | `gemini-3.6-flash` | |
| `chat_answer_temperature` | `0.2` | generation, some variety in phrasing |
| `chat_answer_max_output_tokens` | `1024` | caps output cost |
| `chat_agent_timeout_seconds` | `30` | per agent/grader call, normal stall retries |
| `chat_answer_timeout_seconds` | `90` | the answer call, `attempts=1` |
| `chat_turn_timeout_seconds` | `180` | the whole turn |
| `chat_turn_stale_seconds` | `360` | a `running` turn older than this is treated as abandoned; must exceed the turn timeout (validator) |
| `chat_max_retrieval_rounds` | `2` | loop cap |
| `chat_max_tool_calls_per_round` | `3` | extra parallel tool calls are dropped |
| `chat_history_turns` | `6` | question/answer pairs sent to the agent and answer models |
| `chat_max_question_chars` | `2000` | |
| `chat_max_turns_per_thread` | `100` | bounds checkpoint growth |
| `chat_daily_token_budget` | `500000` | per user, per UTC day, input + output across all chat models |
| `retrieval_bm25_k` | `20` | |
| `retrieval_dense_k` | `20` | |
| `retrieval_rrf_k` | `60` | |
| `retrieval_fused_children` | `8` | |
| `retrieval_max_parents` | `5` | per search call |
| `retrieval_max_context_sources` | `10` | chunk sources across all searches of a turn passed to grade/answer |
| `retrieval_community_k` | `3` | |
| `retrieval_community_min_similarity` | `0.5` | `1 - cosine distance`; calibrate with the golden set |
| `retrieval_hnsw_ef_search` | `100` | |
| `database_echo` | `None` | `bool \| None`. `None` follows `environment == "development"`; an explicit value wins. `.env.example` ships `DATABASE_ECHO=false`, because with chat every question and query vector would otherwise be logged |

Retrieval-quality statement (required by `.claude/rules/llm-pipeline.md`):

- **What it does:** these settings define the new BM25/dense fusion and context size;
  there is no previous behaviour to regress from.
- **Expected impact:** RRF with k=60 favours chunks that rank well in both lists, and
  needs no score normalization between unbounded BM25 scores and cosine distances.
  Five parents of about 1.5k characters each keep the context near 2k tokens per search.
- **How it's verified:** the golden-set hit@5 test (see Test plan) plus inspecting
  retrieved sources in Langfuse traces.

### Shared embedding model (`ai/embeddings.py`)

```python
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768  # must match Vector(768) in models/document.py and models/graph.py
embeddings_model = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL, output_dimensionality=EMBEDDING_DIMENSIONS
)
```

`ai/document_ingestion.py` (`embeddings_model`) and `ai/community_detection.py`
(`_embeddings_model`) import this instance instead of building their own. The reason:
queries must be embedded with exactly the model and dimensions used at indexing time,
so one definition is safer than three. Same model and dimensions, so there is no
behaviour change and no re-embedding. Update the file list in
`.claude/rules/llm-pipeline.md`.

### `ai/requests.py`

- `ai_request(call, *, deadline_seconds=None, attempts=None)`: `attempts=None` keeps
  `settings.ai_request_attempts`. The answer call passes `attempts=1`: once tokens have
  been streamed to the client, resending the call would duplicate text.
- `embed_query(model: Embeddings, text: str) -> list[float]`: raises `ValueError` on
  blank text (Gemini rejects empty parts). It calls `model.aembed_query` under
  `ai_request` with `embedding_request_timeout_seconds`. That method uses task type
  `RETRIEVAL_QUERY`, the asymmetric counterpart of `RETRIEVAL_DOCUMENT`, which
  `aembed_documents` used when the chunks were indexed. It checks
  `len(result) == EMBEDDING_DIMENSIONS`.
- `embed_texts` and `embed_query` wrap each provider call in
  `tracing.embedding_observation(model_name, count)`. LangChain embeddings emit no
  callbacks, so they need manual spans.

### Tracing (`core/tracing.py`)

```python
def tracing_enabled() -> bool: ...
def langchain_callbacks() -> list[BaseCallbackHandler]: ...  # [CallbackHandler()] or []
@contextmanager
def trace_scope(
    name: str, *, session_id: str, user_id: str | None = None,
    tags: list[str], metadata: dict[str, str],
) -> Iterator[None]: ...
@contextmanager
def embedding_observation(model: str, count: int) -> Iterator[None]: ...
def record_event(name: str, metadata: dict[str, str]) -> None: ...
def shutdown_tracing() -> None: ...
```

- **Enabled** only when both keys are set. When disabled, every helper is a no-op and no
  Langfuse client is constructed; CI and tests run like this. Log one INFO line at
  startup saying whether tracing is on.
- **The client** is built explicitly from `Settings`:
  `Langfuse(public_key=…, secret_key=…, base_url=…, environment=settings.environment)`.
  Don't rely on the SDK reading `os.environ`; settings come from `.env` through
  pydantic-settings.
- **`trace_scope`** = `start_as_current_observation(name=…, as_type="span")` combined
  with `propagate_attributes(session_id=…, user_id=…, tags=…, metadata=…)`. That is the
  v4 replacement for `update_current_trace`. In v4, ids are strings of at most 200
  characters and metadata values are strings of at most 200 characters. Check the
  installed SDK's signatures before coding.
- **Prefork safety:** Celery imports task modules in the parent and then forks. The
  OpenTelemetry batch export thread does not survive a fork. The client is created
  lazily in the process that uses it and keyed by PID, the same pattern as
  `tasks/async_runner.get_worker_runner`. `worker_process_shutdown` calls
  `shutdown_tracing()` (add it to the handler in `tasks/async_runner.py`). The API
  lifespan calls it on exit.
- **Fire-and-forget:** export runs on a background thread. A Langfuse outage or a wrong
  base URL must never fail or slow a turn or an ingestion. This is verified manually
  (see End-to-end verification).
- **Span export:** SDK v4 exports only Langfuse-created and `gen_ai.*` spans by default.
  The LangChain `CallbackHandler` spans are Langfuse-created, so no
  `should_export_span` override is needed.

**Ingestion integration** (no prompt or model change, so checkpoint keys stay the same
and no checkpoints are invalidated):

- `tasks/document_ingestion._run_attempt` gets an `attempt: int` parameter
  (`self.request.retries + 1`). It wraps `ingest_document` in
  `trace_scope("document_ingestion", session_id=f"document-{document_id}",
  tags=["ingestion"], metadata={"attempt": …, "final_attempt": …})`. This has to happen
  **inside** the coroutine, because the coroutine runs on the worker's persistent event
  loop and context variables set in the Celery thread would not reach it.
- `ai/graph_extraction.py` and `ai/community_detection.py` pass
  `config={"callbacks": langchain_callbacks(), "run_name": "graph_extraction" |
  "community_summary"}` to `ainvoke` inside the existing `partial(...)`.
- `AttemptCheckpoints.get` calls `record_event("checkpoint_hit", {"kind": kind})` on a
  hit, so a retried attempt shows which results were reused rather than re-requested.

### Retrieval (`financial_assistant/retrieval/`)

**`types.py`**

```python
@dataclass(frozen=True)
class RetrievalScope:
    user_id: int
    document_ids: tuple[int, ...] | None = None   # thread filter
    company_ticker: str | None = None             # thread filter
    document_type: DocumentType | None = None     # agent narrowing
    year: int | None = None                       # agent narrowing

    def narrowed(self, *, company_ticker, document_type, year) -> "RetrievalScope": ...
    # Intersects with existing filters, never replaces them: a thread scoped to NVDA
    # asked for TSLA yields an empty scope, not a TSLA scope.

class ChunkSource(BaseModel):
    kind: Literal["chunk"] = "chunk"
    parent_chunk_id: int
    document_id: int
    filename: str
    company_ticker: str
    document_type: DocumentType
    year: int
    content: str
    score: float          # RRF score of its best child

class CommunitySource(BaseModel):
    kind: Literal["community"] = "community"
    community_id: int
    document_id: int
    filename: str
    company_ticker: str
    title: str
    summary: str
    similarity: float

Source = Annotated[ChunkSource | CommunitySource, Field(discriminator="kind")]
# Later: EntitySource, WebSource.

def source_key(source: Source) -> tuple[str, int]: ...   # dedup key

class Retriever(Protocol):
    name: str
    async def search(
        self, session: AsyncSession, query: str,
        query_embedding: list[float], scope: RetrievalScope,
    ) -> list[Source]: ...
```

**`scope.py`**: `accessible_documents(scope: RetrievalScope) -> Select[tuple[int]]`. It
is the **only** place that encodes access control:

- `document.status = 'completed'`;
- `owner_id IS NULL OR owner_id = :user_id`;
- the optional `id = ANY(:document_ids)`;
- `upper(company_ticker) = upper(:ticker)`, `document_type`, `year`.

Every retriever filters through it.

**`fusion.py`**: `reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], *, k: int)
-> list[tuple[int, float]]`. The score is Σ 1/(k + rank) with 1-based ranks. Ties are
broken by first appearance, so the result is deterministic. It's a pure function.

**`chunks.py`**: `ChunkRetriever`:

1. **BM25 ranking.** An empty or whitespace query skips it.

   ```sql
   SELECT c.id, pdb.score(c.id) AS score
   FROM child_chunk c
   JOIN parent_chunk p ON p.id = c.parent_id
   WHERE c.content ||| :query
     AND p.document_id IN (<accessible_documents>)
   ORDER BY score DESC
   LIMIT :bm25_k
   ```

   `|||` (match disjunction) tokenizes the bound text with the field's tokenizer and
   doesn't parse query syntax. Verified on the dev DB (ParadeDB 0.23.4):
   - a bound parameter works;
   - `what's the "EPS" (diluted): AND -guidance*` runs without error;
   - the plan uses `Custom Scan (ParadeDB Base Scan)` on `child_chunk_bm25_idx`;
   - it takes 4.5 ms with the joins.

2. **Dense ranking**: `ORDER BY c.embedding <=> :query_embedding LIMIT :dense_k`, with
   the same filter. `<=>` matches the new `vector_cosine_ops` HNSW index. Run
   `SET LOCAL hnsw.ef_search = :ef_search` and
   `SET LOCAL hnsw.iterative_scan = relaxed_order` first (pgvector 0.8.1 is installed).
   Filtering on joined document rows can otherwise return fewer than k rows from a plain
   HNSW scan. With `relaxed_order`, re-sort by distance in an outer query. Today an exact
   scan over 4444 rows takes 40 ms, so the planner may keep choosing a sequential scan.
   That's fine; record the plan.
3. Fuse with RRF, keep `retrieval_fused_children` children, map them to parents in
   best-child order, deduplicate, and cap at `retrieval_max_parents`. Load parent
   content plus document metadata in **one** query.

**`communities.py`**: `CommunityRetriever`. Dense search over `graph_community` where
`embedding IS NOT NULL` and `document_id IN (<accessible_documents>)`, top
`retrieval_community_k`. It keeps only results with
`1 - distance >= retrieval_community_min_similarity`.

**`hybrid.py`**: `HybridRetriever(retrievers: Sequence[Retriever], embeddings:
Embeddings)`:

- `search(session, query, scope) -> list[Source]` embeds the query **once**
  (`embed_query`) and runs the retrievers **sequentially** on the given session, because
  one `AsyncSession` can't run concurrent queries.
- It returns chunk sources followed by community sources.
- `default_retriever()` builds it with `ChunkRetriever`, `CommunityRetriever` and
  `ai.embeddings.embeddings_model`.
- A future `EntityRetriever` is appended here.

### Chat agent (`financial_assistant/agents/chat/`)

**`state.py`**

```python
class ChatState(TypedDict):
    # Conversation history: exactly one HumanMessage + one AIMessage per completed turn.
    messages: Annotated[list[AnyMessage], add_messages]
    # Per-turn working memory: agent tool calls, ToolMessages, grader feedback.
    scratchpad: Annotated[list[AnyMessage], add_messages]
    question: str
    # Deduplicated by source_key; `ref` (1-based, order of first appearance) is
    # assigned by the reducer, so it is deterministic for parallel tool calls.
    sources: Annotated[list[NumberedSource], merge_sources]
    round: int
    grade: RetrievalGrade | None
    # Sources cited by each answer, keyed by the AIMessage id (= turn id), for history.
    cited_sources: Annotated[dict[str, list[NumberedSource]], merge_dicts]
```

`NumberedSource` = `{"ref": int, "source": Source}`. `merge_sources` accepts LangGraph's
`Overwrite([])` for resets. Nodes return partial updates and never mutate `state`.

**`context.py`**: `ChatContext` (a frozen dataclass, passed as LangGraph runtime context
and never checkpointed): `user_id`, `thread_id`, `turn_id`, `scope: RetrievalScope`,
`scope_documents` (metadata of at most 50 in-scope documents for the agent prompt),
`session_factory`, `retriever`, `settings`. Tools and nodes read it through
`runtime.context`. That keeps the user and filters out of the LLM's tool arguments.

**`models.py`**

```python
@dataclass(frozen=True)
class ChatModels:
    agent: BaseChatModel                         # tools are bound in the agent node
    grader: Runnable[LanguageModelInput, RetrievalGrade]
    answer: Runnable[LanguageModelInput, dict]   # streams partial JSON dicts

def create_chat_models(settings: Settings) -> ChatModels: ...
```

Each model is a `ChatGoogleGenerativeAI` with `max_retries=1`, the same as the existing
ingestion models. The agent and grader use temperature 0 and `thinking_level` at the
lowest value the model accepts (verify the allowed values in the `llm` smoke test).

- `grader = grader_llm.with_structured_output(RetrievalGrade)`.
- `answer = answer_llm.with_structured_output(<GeneratedAnswer JSON schema dict>,
  method="json_schema")`. With a dict schema LangChain uses `JsonOutputParser`, which
  yields growing partial dicts while streaming. The final dict is validated with
  `GeneratedAnswer.model_validate`.

Tests inject fakes through this dataclass.

**`prompts.py`** holds every prompt in one module, per the llm-engineering skill:

- `AGENT_SYSTEM_PROMPT`: financial research assistant.
  - Must call `search_documents` for any factual question.
  - May narrow by ticker, document type or year.
  - Lists the in-scope documents as data.
- `GRADER_PROMPT`: decides whether the numbered sources contain what the question needs.
  It returns `RetrievalGrade`.
- `ANSWER_SYSTEM_PROMPT`:
  - Answer only from the sources.
  - Cite `[n]` inline and list the cited refs.
  - Say plainly what could not be found, especially when the grade was insufficient at
    the round cap.
- `render_sources(sources) -> str`: wraps each source in
  `<source ref="n" kind="chunk" document="NVDA 10-K 2026 (file.pdf)">…</source>` inside
  `<sources>…</sources>`. Any `</source` or `<source` sequence in the content or
  filename is escaped.

Every prompt states that the text inside `<sources>` and the document listing is
untrusted data and that instructions found there must be ignored. Filenames and PDF text
are user-controlled.

Structured outputs:

```python
class RetrievalGrade(BaseModel):
    sufficient: bool = Field(description="True if the sources contain what is needed to answer")
    missing: str | None = Field(default=None, description="If not sufficient: the missing information, one sentence")

class GeneratedAnswer(BaseModel):
    answer: str = Field(description="Markdown answer with inline [n] citations")  # first property, so it streams first
    cited_refs: list[int] = Field(description="Refs of the sources the answer relies on")
```

**`tools.py`**

```python
class SearchDocumentsArgs(BaseModel):
    query: str = Field(min_length=2, max_length=300, description="…")
    company_ticker: str | None = Field(default=None, description="Narrow to one company")
    document_type: DocumentType | None = None
    year: int | None = None

@tool("search_documents", args_schema=SearchDocumentsArgs)
async def search_documents(..., runtime: ToolRuntime[ChatContext, ChatState]) -> Command: ...

CHAT_TOOLS: list[BaseTool] = [search_documents]   # extension point: web_search goes here
```

- Narrows `runtime.context.scope` (intersection only). An empty intersection returns a
  ToolMessage saying no documents match, without running a query.
- Opens **its own** session from `runtime.context.session_factory`. `ToolNode` runs
  parallel tool calls concurrently, and sessions can't be shared between tasks.
- Writes custom stream events through `runtime.stream_writer`: `status` (searching, with
  the query) and `sources` (newly found sources with their refs).
- Returns `Command(update={"sources": found, "scratchpad": [ToolMessage(<compact
  listing: document + first 200 chars per source>, tool_call_id=runtime.tool_call_id)]})`.
  The agent sees a compact listing, while the grade and answer nodes see the full
  sources.
- A future `web_search` tool returns `WebSource`s through the same `Command` shape; the
  grade and answer prompts render any `Source` kind.
- Errors: invalid arguments raise `ToolInvocationError`, which `ToolNode`'s default
  handler turns into an error ToolMessage for the agent (verify that default in the
  installed langgraph). Database or embedding errors **propagate** and the turn fails
  with `retrieval_unavailable`. Internal errors are never fed back to the LLM.

**`nodes.py`**: each node is a plain async function `(state, runtime) -> partial update`.

- `prepare` resets per-turn fields:
  `{"scratchpad": [RemoveMessage(id=REMOVE_ALL_MESSAGES)], "sources": Overwrite([]),
  "round": 0, "grade": None}`.
- `agent` builds `[System(AGENT_SYSTEM_PROMPT + scope documents)] + last
  chat_history_turns pairs of messages + Human(question) + scratchpad`. It calls
  `models.agent.bind_tools(CHAT_TOOLS, tool_choice="any" if round == 0 else "auto")`
  via `ai_request(..., deadline_seconds=chat_agent_timeout_seconds)`. Stall retries are
  safe here because the agent's output is never streamed to the client. It truncates
  `tool_calls` beyond `chat_max_tool_calls_per_round` (logged) and returns
  `{"scratchpad": [ai_message], "round": round + 1}`.
- `tools` is `ToolNode(CHAT_TOOLS, messages_key="scratchpad")`.
- `grade` renders the question plus at most `retrieval_max_context_sources` chunk
  sources and the community sources. It returns `{"grade": grade}`, plus
  `{"scratchpad": [HumanMessage("Reviewer: … missing: …; search again with a different
  query")]}` when the grade is insufficient. An `OutputParserException` means "treat as
  sufficient" (logged): a broken grade must not fail the turn, and the answer prompt
  already handles missing information.
- `generate` consumes `models.answer.astream(prompt)` inside
  `ai_request(..., deadline_seconds=chat_answer_timeout_seconds, attempts=1)`:
  - For each partial dict it writes `{"type": "token", "text": delta}`, where `delta` is
    the new suffix of `partial["answer"]` (skipped if the partial isn't a prefix
    extension).
  - It validates the final dict and drops `cited_refs` not present in `sources`.
  - It returns `{"messages": [HumanMessage(question, id=f"{turn_id}:q"),
    AIMessage(answer, id=str(turn_id))], "cited_sources": {str(turn_id): cited},
    "scratchpad": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}`.
  - It writes a final `{"type": "answer", ...}` event.
  - An empty or invalid final answer (e.g. a safety block) raises `AnswerUnavailable` →
    error code `answer_unavailable`.
- Routing:
  - `route_after_agent`: if the last scratchpad message has tool calls → `tools`,
    otherwise → `generate`.
  - `route_after_grade`: if sufficient or `round >= chat_max_retrieval_rounds` →
    `generate`, otherwise → `agent`.

**`graph.py`**:
`build_chat_graph(models: ChatModels, *, checkpointer: BaseCheckpointSaver, tools:
Sequence[BaseTool] = CHAT_TOOLS) -> CompiledStateGraph`.

- Edges: `StateGraph(ChatState, context_schema=ChatContext)`, then START → `prepare` →
  `agent` → conditional → `tools` → `grade` → conditional, `generate` → END.
- The loop terminates through `round`. `recursion_limit = 3 * max_rounds + 4` is passed
  per run as a backstop.

**`runner.py`**: `ChatTurnRunner(graph, session_factory, retriever, settings)` with
`stream(*, user_id, thread, turn_id, question) -> AsyncIterator[ChatEvent]`.

- **Its own task.** `stream` starts `asyncio.create_task(self._run(...))`, which pushes
  `ChatEvent`s into an `asyncio.Queue`; `stream` yields from the queue. In its
  `finally` (client disconnect, server shutdown) it calls `task.cancel()`, which is
  synchronous and needs no `await`.
  - *Why a separate task:* FastAPI's SSE producer runs the generator inside anyio cancel
    scopes, where cancellation is re-raised at **every** `await`. Cleanup awaits in the
    generator's `finally` (finalizing `chat_turn`, ending Langfuse spans) would be
    cancelled again.
  - A plain asyncio task receives a single `CancelledError` and can finish its cleanup.
  - Running tasks are kept in a module-level set; the lifespan cancels and awaits them on
    shutdown.
- **`_run`** runs inside `asyncio.timeout(chat_turn_timeout_seconds)` and
  `trace_scope("chat_turn", session_id=str(thread.id), user_id=str(user_id),
  tags=["chat"], metadata={"turn_id": …})`:
  1. Build the scope and load the in-scope document metadata. If no accessible completed
     document is in scope, emit a fixed `answer` ("No processed documents are available
     in this chat's scope…"), finalize `completed` with 0 tokens and return. No model
     calls are made and no checkpoint is written.
  2. Run `graph.astream({"question": question}, config={"configurable": {"thread_id":
     str(thread.id)}, "callbacks": [usage_handler, *langchain_callbacks()],
     "recursion_limit": …, "run_name": "chat_turn"}, context=ChatContext(...),
     stream_mode="custom", durability="exit")` and forward its events. Every SSE event
     comes from the custom stream, so token chunks of the agent/grader calls are never
     sent to the client.
  3. `finish_turn(...)` with the status, `usage_handler.usage_metadata` (per-model input
     and output tokens) and the error code. Update `chat_thread.updated_at` and set the
     title from the first question if the thread has none.
- **`durability="exit"`** writes the state once, when the run ends. `messages` only
  changes in `generate`, the last node, so a failed or cancelled turn never leaves half a
  turn in the history. `prepare` resets the per-turn fields next time, whatever was
  persisted.
- **Error mapping** (the `error` event, then finalize `failed`):

  | Exception | Code |
  | --- | --- |
  | `TimeoutError` | `timeout` |
  | Gemini `ClientError` 408/429 and server errors | `model_unavailable` |
  | Other provider 4xx (`is_permanent_provider_error`) | `model_error` |
  | SQLAlchemy/psycopg errors | `retrieval_unavailable` |
  | `AnswerUnavailable` | `answer_unavailable` |
  | Anything else | `internal` (logged with traceback; generic message) |

  `CancelledError` → finalize `cancelled` with the usage so far, then re-raise.

**`turns.py`** contains the database functions:

```python
class ThreadBusyError(Exception): ...

async def tokens_used_today(session: AsyncSession, user_id: int) -> int: ...
async def claim_turn(session: AsyncSession, *, thread_id: UUID, user_id: int,
                     stale_after: timedelta) -> ChatTurn: ...
async def finish_turn(session_factory: async_sessionmaker[AsyncSession], turn_id: UUID, *,
                      status: ChatTurnStatus, usage: Mapping[str, UsageMetadata],
                      error_code: str | None) -> None: ...
```

- **`claim_turn`** first runs
  `UPDATE chat_turn SET status='failed', error_code='abandoned', finished_at=now()
  WHERE thread_id=:t AND status='running' AND started_at < now() - :stale`. It then
  inserts a `running` row and commits immediately, so no connection is held during the
  turn. An `IntegrityError` on `uix_chat_turn_running_thread` raises `ThreadBusyError`
  (409).
  - *Why not a row lock:* `FOR UPDATE NOWAIT` (the ingestion pattern) would hold a
    pooled connection for the whole 5–60 s stream.
  - The crash-safety it would have given comes from the stale timeout instead.
- **`tokens_used_today`** sums `input_tokens + output_tokens` for the user since UTC
  midnight. It is served by `ix_chat_turn_user_id_started_at`.

### API (`api/routers/chat.py`, `schemas/chat.py`, `api/dependencies/chat.py`)

All endpoints require `get_current_user`. A thread owned by another user is answered
with `404`, exactly like `documents.py`.

| Method & path | Body / params | Response |
| --- | --- | --- |
| `POST /chat/threads` | `{title?: str ≤200, document_ids?: list[int] (1–50, unique), company_ticker?: str}` | `201 ThreadRead`. `404` if any `document_id` is not accessible (no existence leak); documents still processing are allowed and become searchable once completed |
| `GET /chat/threads` | `limit` (1–100, default 50), `offset` | `ThreadRead[]`, newest `updated_at` first |
| `GET /chat/threads/{thread_id}` | | `ThreadDetail`: thread + `messages: [{id, role, content, sources?, created_at}]`, read from `graph.aget_state`; `sources` from `cited_sources`, `created_at` from the matching `chat_turn` |
| `DELETE /chat/threads/{thread_id}` | | `204`. `409` while a turn is running. Deletes the checkpoints first (`checkpointer.adelete_thread`), then the row, so a failed DELETE can be retried; `chat_turn` rows survive (`thread_id` → NULL) to keep budget accounting |
| `POST /chat/threads/{thread_id}/messages` | `{content: str}` | `text/event-stream` |

The message endpoint does everything that can fail with a proper HTTP status **before**
the first byte. After that, failures arrive as `error` events.

1. `422`: `content` is empty after stripping, or longer than `chat_max_question_chars`.
2. `404`: the thread is missing or owned by someone else.
3. `409 thread_full`: the thread already has `chat_max_turns_per_thread` completed
   turns.
4. `429`: `tokens_used_today >= chat_daily_token_budget`. `Retry-After` gives the
   seconds until the next UTC midnight.
5. `409 thread_busy`: `claim_turn` raised `ThreadBusyError`.
6. Return the generator (`response_class=EventSourceResponse`), which yields
   `ServerSentEvent(event=…, data=<pydantic model>)` from `runner.stream(...)`. FastAPI
   sends a `: ping` keep-alive every 15 s on its own.

SSE events (payload models in `schemas/chat.py`):

| Event | Data | When |
| --- | --- | --- |
| `turn` | `{turn_id}` | First event |
| `status` | `{stage: "searching" \| "grading" \| "answering", query?: str}` | Progress |
| `sources` | `{sources: [SourceRead (ref, kind, document_id, filename, company_ticker, document_type, year, excerpt ≤300 chars, title?)]}` | Each time a search adds sources |
| `token` | `{text}` | Answer deltas |
| `done` | `{turn_id, message: {id, content, citations: [ref], sources: [SourceRead]}, usage: {input_tokens, output_tokens}}` | Success; the last event |
| `error` | `{code, message}` | Failure; the last event |

`api/dependencies/chat.py` defines `get_chat_runner(request)` and
`get_checkpointer(request)`, both read from `app.state`, so tests can override them.

**Lifespan (`api/server.py`):**

- **Startup:** `init_langgraph_pool()` → `AsyncPostgresSaver(conn=pool)` →
  `build_chat_graph(create_chat_models(settings), checkpointer=…)` →
  `ChatTurnRunner(graph, async_session_maker, default_retriever(), settings)`, all stored
  on `app.state`. Tracing initializes lazily.
- **Shutdown:** cancel and await the running turns, `shutdown_tracing()`, close the pool.

**`core/db.py`:** `create_async_engine(database_url, echo=settings.database_echo_enabled)`
replaces the hard-coded `echo=True`. `database_echo_enabled` is a property on `Settings`
(`database_echo` if set, else `environment == "development"`), unit-tested for both
branches and the override. Also add `row_factory=dict_row` and `prepare_threshold=0` to
the LangGraph pool's `kwargs`. `AsyncPostgresSaver` reads rows by column name and documents both for
user-supplied connections; the default tuple rows raise `TypeError`. `setup()` is not
called here.

**Account deletion (`core/accounts.delete_account`, from the prerequisite fix):** a user
who owns documents is now tombstoned rather than deleted, so the `ON DELETE CASCADE` on
`chat_thread.user_id` and `chat_turn.user_id` does **not** fire for them. `delete_account`
therefore deletes chat data explicitly, for hard-deleted and tombstoned users alike: call
`checkpointer.adelete_thread` for each of the user's threads, then delete their
`chat_turn` and `chat_thread` rows. The checkpointer is optional (`checkpointer=None`
skips it) and is passed in by `DELETE /auth/me` from `app.state`. Order: checkpoints
first, so a failure leaves the rows and a retry can finish the job.

**`services/setup_checkpointer.py`:** `python -m
financial_assistant.services.setup_checkpointer` opens
`AsyncPostgresSaver.from_conn_string(<libpq URL>)` and runs `await saver.setup()`.

- *Why not Alembic:* the saver's own migrations include `CREATE INDEX CONCURRENTLY`,
  which can't run inside Alembic's transaction. It also tracks its own versions in
  `checkpoint_migrations`; `alembic/env.py` already ignores the four `checkpoint*`
  tables.
- *Why not the lifespan:* several uvicorn workers would race on `CREATE TABLE`. The
  compose `migrate` service runs once, before the app and the worker.

The compose `migrate` command becomes `["/bin/sh", "-c", "/opt/venv/bin/alembic
upgrade head && /opt/venv/bin/python -m
financial_assistant.services.setup_checkpointer"]`. The libpq URL conversion
(`postgresql+psycopg://` → `postgresql://`), currently inline in `init_langgraph_pool`,
moves to a small helper in `core/db.py` that both use.

### Documentation to update

- **`CLAUDE.md`:**
  - Architecture: `retrieval/`, `agents/chat/`, the chat router, `core/tracing.py`,
    `ai/embeddings.py`.
  - Commands: `uv run python -m financial_assistant.services.setup_checkpointer` after
    `alembic upgrade head`, and the golden-set command.
  - Replace the "Not implemented yet" line.
- **`.claude/rules/llm-pipeline.md`:**
  - The embedding model now lives in `ai/embeddings.py`.
  - The fusion settings now exist (list their names and defaults).
  - `retrieval/` and `agents/` are no longer "planned".
  - Chat model calls go through `ai_request` too.
- **`.claude/rules/database.md`:** an HNSW index exists (`ix_child_chunk_embedding`,
  `vector_cosine_ops`, so queries must use `<=>`). The checkpointer tables are created
  by `setup_checkpointer`, not Alembic.

## Data & migrations

**Migration 1: `add_chat_tables`** (autogenerate, then review)

`chat_thread`:

| Column | Type | Notes |
| --- | --- | --- |
| `id` | `uuid` PK, default `uuid4` | Also the LangGraph `thread_id` |
| `user_id` | `int` FK `user.id` `ON DELETE CASCADE`, not null | |
| `title` | `varchar(200)` null | Set from the first question (whitespace-collapsed, 80 chars) if not given |
| `document_ids` | `int[]` null | Thread filter. No FK: a deleted document simply drops out of `accessible_documents` |
| `company_ticker` | `varchar` null | Thread filter |
| `created_at`, `updated_at` | `timestamptz` server default `now()` | `updated_at` is set by `finish_turn` |

Index: `ix_chat_thread_user_id_updated_at (user_id, updated_at DESC)` for listing.

`chat_turn`:

| Column | Type | Notes |
| --- | --- | --- |
| `id` | `uuid` PK | Also the AIMessage id |
| `thread_id` | `uuid` FK `chat_thread.id` `ON DELETE SET NULL`, null | Deleting a thread must not reset the budget |
| `user_id` | `int` FK `user.id` `ON DELETE CASCADE`, not null | |
| `status` | `chat_turn_status_enum` (`running`, `completed`, `failed`, `cancelled`) | `values_callable` like the other enums |
| `error_code` | `varchar(50)` null | Code only, never provider messages |
| `input_tokens`, `output_tokens` | `int` not null, server default `0` | Totals across models |
| `usage` | `jsonb` not null, server default `'{}'` | Per-model `UsageMetadata` |
| `started_at` | `timestamptz` server default `now()` | |
| `finished_at` | `timestamptz` null | |

Indexes:
- `ix_chat_turn_user_id_started_at (user_id, started_at)` serves the budget query.
- The partial unique index `uix_chat_turn_running_thread (thread_id) WHERE status =
  'running'` is the concurrency guard. Declare it on the model with
  `postgresql_where`, so autogenerate sees it; check that the generated migration
  contains the `WHERE`.

Downgrade drops both tables and `chat_turn_status_enum`.

**Migration 2: `add_hnsw_index_to_child_chunk`**

Declared on `ChildChunk.__table_args__` so autogenerate and the `create_all` schemas in
tests see it:
`Index("ix_child_chunk_embedding", "embedding", postgresql_using="hnsw",
postgresql_with={"m": 16, "ef_construction": 64}, postgresql_ops={"embedding":
"vector_cosine_ops"})`.

- **Locking:** a plain `CREATE INDEX` takes a SHARE lock, which blocks ingestion commits
  into `child_chunk` while the index builds. That's under a second at 4.4k rows. At
  large scale, switch to `CONCURRENTLY` inside `op.get_context().autocommit_block()`.
- **Downgrade:** `DROP INDEX IF EXISTS ix_child_chunk_embedding`.
- The dimension stays at 768 and no data changes.

**LangGraph checkpointer tables** (`checkpoints`, `checkpoint_blobs`,
`checkpoint_writes`, `checkpoint_migrations`) are created by `setup_checkpointer`. With
`durability="exit"` there is one checkpoint per turn. The `messages` channel is stored as
a new blob version every turn, so storage per thread grows roughly quadratically with
turn count; `chat_max_turns_per_thread=100` bounds it.

**Tests:** `migrated_database` also runs `setup_checkpointer` after `alembic upgrade
head` (subprocess, same env). `private_database` additionally creates the BM25 index
inside its private schema, with the DDL of migration `95900fffcf36` kept as a constant in
`tests/fixtures/database.py` and a comment pointing at the migration. This lets
retrieval tests use it (the earlier note "nothing tests BM25 yet" no longer holds). After
switching branches, the usual `DROP DATABASE "<POSTGRES_DB>_test" WITH (FORCE)` recovery
applies.

No re-embedding, no backfill and no existing data to migrate.

## External calls & cost

| Call | Model | Per turn | Input tokens | Output tokens |
| --- | --- | --- | --- | --- |
| Agent | `gemini-3.5-flash-lite` | 1–2 | ~2k (round 1), ~3k (round 2) | ~50 (tool calls) |
| Query embedding | `models/gemini-embedding-001` | 1–3 per round | ~20 | – |
| Grader | `gemini-3.5-flash-lite` | 1–2 | ~2.5k (round 1), ~4.5k (round 2) | ~50 |
| Answer | `gemini-3.6-flash` | 1 | ~4.5k (1 round), ~6.5k (2 rounds) | ~400 (cap 1024) |

**Prices** are per million tokens, from aggregators rather than Google (2026-10-14):
- flash-lite: $0.30 in / $2.50 out;
- flash: $1.50 / $7.50 from 2027-01-01, with an introductory $0.75 / $3.75 until
  2026-12-31.

Confirm both at <https://ai.google.dev/gemini-api/docs/pricing> before relying on them.

**Per-turn estimate** (post-intro prices):
- a typical one-round turn costs about $0.011 (lite about $0.0016 + flash about $0.0098);
- a two-round turn costs about $0.017;
- the 500k-token daily budget therefore allows about 45–90 turns, roughly ≤ $1 per user
  per day.

**Thinking tokens:** Gemini 3.x models think by default. Thinking tokens are billed as
output and count against the budget. Verify in the smoke test that `usage_metadata`
includes them. This is why the agent and grader run at the lowest thinking level.

**Cost controls:** question length cap, sliding history window, per-search parent cap,
context-source cap, output-token cap, round cap, tool calls per round cap, daily budget,
and no model calls when no documents are in scope.

**Ingestion:** tracing adds no model calls; prompts, models and checkpoint keys are
unchanged.

**Langfuse** (no model cost):
- a chat turn produces about 15–25 observations (LangGraph node and runnable spans plus
  embedding spans);
- an average ingestion produces about 150 or more (88 extraction calls, 31 summaries, 28
  embedding batches for the current corpus average, plus chain spans).

Check the Langfuse Cloud plan's monthly quota.

**Golden-set `llm` run:**
- ingesting a ~10-page PDF (~15 extraction calls, ~5 summaries, ~6 embedding batches)
  costs about $0.03;
- 12 chat turns cost about $0.20;
- the smoke test costs about $0.01.

That is under $0.30 per run, and it never runs in CI.

## Edge cases

**Scope and access**
- **Another user's private document in `document_ids`:** `404` at thread creation.
  Public documents are allowed.
- **A document deleted after thread creation:** it silently leaves the scope. History
  keeps the cited-source snapshots (the excerpt in `cited_sources`).
- **Nothing in scope** (no accessible completed documents): fixed answer, no model
  calls, turn `completed` with 0 tokens.
- **Agent narrowing outside the thread scope** (thread on NVDA, agent asks for TSLA):
  empty intersection, a "no documents match" ToolMessage, no query.
- **A document still `processing`** is excluded until it completes, as is a `failed`
  document.
- **Ticker case** (`nvda` vs `NVDA`) is compared case-insensitively.

**Retrieval**
- **BM25 special characters:** `|||` tokenizes, so quotes, `:`, `AND`, `-`, `*` and
  parentheses don't error (verified on the dev DB); integration-tested.
- **No BM25 hits:** RRF runs over the dense list alone.
- **Irrelevant results:** dense search always returns k rows, even for chit-chat or
  irrelevant questions. The grader marks the result insufficient, round 2 runs, and the
  answer states that the information was not found. "Thanks!" therefore costs two cheap
  rounds, the accepted price of forced search.
- **Duplicate sources** across rounds or parallel calls are deduplicated by
  `source_key`; refs stay stable within a turn.
- **Community similarity below the floor:** no community sources, which is normal.
- **HNSW plus filters** can return fewer than k rows without iterative scan, so
  `relaxed_order` is set and the results are re-sorted.

**Agent and LLM**
- **More than 3 parallel tool calls:** the extra calls are dropped (logged).
- **Invalid tool arguments:** an error ToolMessage goes back to the agent. The call
  still counts toward the round, and the round cap holds.
- **Round 2 with no tool calls:** goes to `generate`.
- **Unparseable grader output:** treated as sufficient (logged).
- **Answer JSON:** cited refs not in `sources` are dropped. An empty answer or a safety
  block raises `answer_unavailable`.
- **A partial `answer` that isn't a prefix extension** (a parser artefact): the delta is
  skipped and the final validated text is authoritative in `done`.
- **Prompt injection** via PDF text or filenames: the content is delimited and escaped,
  the tools are read-only, and scope is enforced server-side. At worst an injected
  document skews the answer; it can't widen access.
- **Gemini 429/quota:** one LangChain retry (`max_retries=1`), then `model_unavailable`.
  A permanent 4xx raises `model_error`.
- **Stalls:** agent and grader calls get the normal stall retries. The answer call gets
  no retry (`attempts=1`) and fails with `timeout`. The whole turn is capped at 180 s.

**Turns and concurrency**
- **Two messages on the same thread at once:** the second gets `409 thread_busy`
  (partial unique index). Turns on different threads of the same user are allowed;
  in-flight turns can push the budget over by their size (documented, accepted).
- **Client disconnect mid-stream:** the run is cancelled, the turn is `cancelled` with
  the usage so far, and the history is unchanged.
- **API crash mid-turn:** the `running` row stays until `chat_turn_stale_seconds` (360
  s), then the next claim marks it `failed/abandoned`. Until then the thread answers
  `409` with `Retry-After`.
- **DELETE on a thread with a running turn:** `409`.
- **Budget day boundary:** UTC midnight, with `Retry-After` in seconds.
- **Long threads:** prompts use the last 6 pairs; at 100 turns the thread answers `409
  thread_full`.
- **Account deletion:** `delete_account` deletes the user's checkpoints, turns and threads
  explicitly (the cascade does not fire for tombstoned users).
- **Documents of a deleted account** keep their `owner_id` (prerequisite fix), so
  `accessible_documents` never returns them to anyone, including from existing threads
  whose `document_ids` mention them, until an admin publishes them.

**Tracing and infrastructure**
- **Langfuse unreachable or slow:** background export; turns and ingestion are
  unaffected.
- **Keys absent** (CI, tests): tracing disabled, no client constructed.
- **Prefork Celery workers:** the client is created per child process (keyed by PID)
  and flushed on `worker_process_shutdown`.
- **Several uvicorn workers:** each has its own LangGraph pool (`max_size=20`) and
  Langfuse client. The SQLAlchemy pool (default 5 + 10 overflow) bounds concurrent
  retrieval, since each tool call holds one session briefly.
- **SQL echo:** the engine used to hard-code `echo=True`, which would log every question
  and a 768-float vector per retrieval query. It is now the `database_echo` setting, and
  `.env.example` ships it as `false`. Anyone who enables it logs user questions.

## Test plan

Implement in phases, each ending with format, lint, type check and fast tests green.

0. **Prerequisite (separate branch, already merged):** the orphaned-documents fix.
   Verify `core/accounts.py` and `User.deleted_at` exist before starting.
1. **Dependencies:** `uv add`, record versions, run the fast suite.
2. **Embeddings, requests and tracing:** `ai/embeddings.py`, the `ai/requests.py`
   additions, `core/tracing.py`, ingestion tracing.
3. **Data:** models, migrations, `setup_checkpointer`, fixture changes.
4. **Retrieval:** `retrieval/`.
5. **Agent:** `agents/chat/` (state, nodes, graph, runner, turns).
6. **API and lifespan.**
7. **`llm` tests and docs.**

Per the testing rules, nodes are tested as plain functions before the compiled graph.

**Unit** (`tests/unit/…`, `pytestmark = pytest.mark.unit`, fakes only):

- `ai/test_requests.py` (extend): `attempts=1` doesn't retry a stall; `embed_query`
  rejects blank text, calls `aembed_query` under the embedding deadline and checks the
  dimension.
- `core/test_tracing.py`:
  - disabled without keys: `langchain_callbacks() == []`, scopes are no-ops, no client is
    constructed;
  - enabled: the client is built from settings (Langfuse class mocked);
  - a PID change rebuilds the client;
  - `shutdown_tracing` is idempotent.
- `retrieval/test_fusion.py`: RRF scores for known rankings; one empty list; both empty;
  ties broken by first appearance; a larger k flattens differences.
- `retrieval/test_types.py`: `RetrievalScope.narrowed` intersects filters and never
  widens; conflicting ticker → empty.
- `retrieval/test_hybrid.py`: with fake retrievers and `DeterministicFakeEmbedding(size=768)`
  the query is embedded once, retrievers run in order, and a blank query is rejected.
- `agents/chat/test_state.py`: `merge_sources` deduplicates and assigns stable refs;
  `Overwrite([])` resets; `merge_dicts`.
- `agents/chat/test_prompts.py`: `render_sources` escapes `</source>` in content and
  filenames; history trimming keeps the last N pairs.
- `agents/chat/test_nodes.py`:
  - `prepare` resets;
  - `agent` binds tools with `tool_choice="any"` in round 1 and `"auto"` in round 2, and
    truncates extra tool calls;
  - `route_after_agent` and `route_after_grade` (sufficient, insufficient below the cap,
    cap reached);
  - `grade` falls back on a parse error;
  - `generate` emits prefix deltas, drops unknown refs, appends exactly one Human + AI
    pair with the turn-id ids, clears the scratchpad, and raises `AnswerUnavailable` on
    an empty answer.

  Fakes: add `FakeToolCallingChatModel` to `tests/utils.py`. `GenericFakeChatModel`
  doesn't implement `bind_tools`, so the fake records the bound tools and `tool_choice`
  and replays scripted `AIMessage`s. Grader and answer fakes are `RunnableLambda` /
  `RunnableGenerator` objects yielding scripted values.
- `agents/chat/test_graph.py` (compiled graph, `InMemorySaver`, fake retriever tool
  context):
  - one full turn → `messages` has 2 entries;
  - a second turn sees the first in the agent prompt;
  - an always-insufficient grader stops after exactly 2 rounds;
  - custom stream events arrive in order (status, sources, token…, answer);
  - a run that raises inside `generate` leaves the previous turn's `messages` unchanged.
- `agents/chat/test_runner.py` (fake graph and an in-memory turn store):
  - the event sequence and the `done` payload;
  - `TimeoutError` → `error/timeout` and finalize `failed`;
  - a 429 `provider_error` → `model_unavailable`;
  - cancelling the consumer cancels the task and finalizes `cancelled` with partial
    usage;
  - no documents in scope → fixed answer with no graph run.
- `api/routers/test_chat.py`: request-schema validation only (blank or long content,
  more than 50 `document_ids`), with dependencies overridden.

**Integration** (`tests/integration/…`, Postgres):

- `retrieval/test_chunks.py` and `retrieval/test_communities.py`, on `private_database`,
  which now has the BM25 index; vectors come from `DeterministicFakeEmbedding`:
  - BM25 finds the exact-term chunk;
  - dense finds the nearest;
  - fusion favours a chunk ranked in both lists;
  - parents are deduplicated, ordered and capped;
  - another user's private document is never returned, while a public one is; a
    document whose owner is tombstoned (`deleted_at` set) is never returned to anyone;
  - `processing`/`failed` documents are excluded;
  - the `document_ids`, ticker (case-insensitive), type and year filters work;
  - special-character queries don't error;
  - the community similarity floor is applied;
  - `ix_child_chunk_embedding` exists in the schema.
- `agents/chat/test_turns.py` (`private_database`):
  - a second `claim_turn` on the same thread raises `ThreadBusyError`;
  - a stale running turn is marked `abandoned` and the new claim succeeds;
  - `tokens_used_today` counts only today (UTC) and only this user;
  - deleting a thread keeps its turns with `thread_id` NULL.
- `agents/chat/test_checkpointer.py`: a new fixture `langgraph_checkpointer`
  (depends on `migrated_database`, uses the test DB, gets added to `_DB_FIXTURES`):
  - `setup()` is idempotent;
  - the graph with fake models persists state across two runs;
  - `adelete_thread` removes it;
  - the pool's `dict_row` setting works.
- `api/routers/test_chat.py` (through `client`), with `get_chat_runner` and
  `get_checkpointer` overridden (a fake runner, or the real runner with fake models and
  `InMemorySaver`). A new `session_factory` fixture (sessions bound to the test
  connection, `join_transaction_mode="create_savepoint"`) is used for the real runner's
  `finish_turn`. Keep fake agents to one tool call per round here, because the sessions
  share one connection.
  - create thread: inaccessible `document_ids` → 404, public allowed;
  - list returns only own threads, newest first;
  - get thread: another user's → 404, history with sources;
  - delete → 204 and the checkpoints are gone, 409 while running;
  - send message:
    - SSE events arrive in order;
    - 409 `thread_busy` with a running turn;
    - 409 `thread_full`;
    - 429 with `Retry-After` when the budget is exhausted (seeded `chat_turn` rows);
    - 422 for blank content;
    - 401 without a token;
  - `DELETE /auth/me` deletes the user's checkpoints, threads and turns, both for a user
    without documents (hard delete) and for one with documents (tombstone).

**E2E** (`tests/e2e/test_chat.py`, `RUN_INGESTION_TESTS=1`): extend the existing e2e
flow. Upload a PDF, let the worker ingest it with fake models, then send one message via
the API with fake chat models but **real** retrieval over the `private_database` schema.
Assert that a `sources` event references the uploaded document.

**LLM** (`tests/llm/…`, `RUN_LLM_TESTS=1`, never in CI). Add `tests/llm/conftest.py`
importing `migrated_database` and `private_database`.

- `agents/chat/test_models_smoke.py`:
  - the three model IDs respond;
  - `bind_tools(tool_choice="any")` yields a tool call;
  - the grader returns a `RetrievalGrade`;
  - **the answer runnable streams more than one growing partial dict** (this verifies
    the partial-JSON streaming assumption early, in phase 2);
  - `usage_metadata` is present, including thinking tokens;
  - the allowed `thinking_level` values are confirmed.
- `agents/chat/test_golden_set.py`:
  - **Fixture:** `tests/llm/fixtures/northwind/generate_report.py` (run with
    `uv run --with reportlab python …`) generates `northwind_10k.pdf`, about 10 pages of
    a fictional company's 10-K with known figures, segments, executives and risks. The
    PDF is committed. `golden_set.json` holds about 12 questions, each with an expected
    fact and an expected source substring. It includes 2 comparison or multi-hop
    questions and 2 unanswerable ones.
  - **Run:** ingest with the real pipeline (storage path redirected as in the e2e
    flow), then run each question through the real graph with an `InMemorySaver`.
  - **Assertions:**
    - retrieval hit@5 (the expected substring appears in a cited or retrieved source) is
      at least 0.8;
    - at least 8 of 10 answerable questions contain the expected figure and cite a
      source containing it;
    - both unanswerable questions say the information was not found.

  Fictional facts can't come from model memory, so this measures grounding. Print the
  per-question results to use them to calibrate `retrieval_community_min_similarity`.

## End-to-end verification steps

1. `uv add langgraph-checkpoint-postgres langfuse`, then
   `uv pip show langgraph langgraph-checkpoint langgraph-checkpoint-postgres langfuse`,
   then `uv run pytest -m "not llm and not db" -x -q` (green before any feature code).
2. `uv run alembic upgrade head && uv run python -m
   financial_assistant.services.setup_checkpointer`, run twice to confirm it's
   idempotent. Check with `psql`: `\d chat_thread`, `\d chat_turn` (the partial unique
   index shows its `WHERE`), `\di ix_child_chunk_embedding` (hnsw, `vector_cosine_ops`),
   and `\dt checkpoint*`. Then `uv run alembic downgrade -1` twice and
   `uv run alembic upgrade head` again, on the dev DB, after asking (it drops the new
   tables).
3. Plans: `EXPLAIN ANALYZE` the BM25 and dense queries as generated by the code, with
   real parameters. Report whether `child_chunk_bm25_idx` and `ix_child_chunk_embedding`
   are used and the execution times. A sequential scan for dense search is acceptable at
   4.4k rows; record it.
4. Start the API
   (`uv run uvicorn financial_assistant.api.server:app --reload --host 0.0.0.0 --port 8000`).
   Then `POST /auth/register`, `POST /auth/token`, `POST /chat/threads` with
   `{"company_ticker": "NVDA"}`, then
   `curl -N -H "Authorization: Bearer $T" -H 'Content-Type: application/json' -d
   '{"content": "What was NVIDIA data center revenue?"}'
   localhost:8000/chat/threads/$ID/messages`. You should see `turn`, `status`,
   `sources`, `token`… and `done`. Ask a follow-up ("And how did it change vs last
   year?") and check that it's resolved from history. `GET /chat/threads/$ID` shows both
   turns with sources.
5. Failure paths:
   - two concurrent `curl`s on the same thread → the second gets `409`;
   - Ctrl-C mid-stream → the `chat_turn` row is `cancelled` and history is unchanged;
   - `CHAT_DAILY_TOKEN_BUDGET=1` → `429` with `Retry-After`;
   - a thread on a ticker with no documents → fixed answer, 0 tokens;
   - `DELETE /chat/threads/$ID` → `204`, and its rows in `checkpoints` are gone.
6. Langfuse (keys in `.env`): the chat trace has session = thread id, user = user id,
   tag `chat`, spans for `prepare/agent/tools/grade/generate`, embedding spans, and token
   usage per model. Upload a small PDF with `docker compose up -d` running: one
   `document_ingestion` trace with session `document-<id>`, extraction, summary and
   embedding observations. Set `LANGFUSE_BASE_URL=https://invalid.example` and repeat a
   turn: it still succeeds, with no added latency beyond the warning logs.
7. `RUN_LLM_TESTS=1 uv run pytest tests/llm -m llm -x -q -s` (smoke test, then golden
   set). Record hit@5 and the answer score.
8. Quality gates: `uv run ruff check --fix . && uv run ruff format .`,
   `uv run pyright`, `uv run pytest -m "not llm and not db" -x -q`,
   `uv run pytest -m integration -x -q` (with `RUN_RABBITMQ_TESTS=1`), and
   `RUN_INGESTION_TESTS=1 uv run pytest -m e2e -x -q`.

## Open questions

1. **Model IDs and prices.** `gemini-3.5-flash-lite` and `gemini-3.6-flash` come from
   the existing code and aggregator sites. The smoke test confirms the IDs; confirm the
   prices on Google's pricing page. Sources disagree about Flash's intro vs.
   post-intro rate.
2. **Partial-JSON streaming of the structured answer** (`with_structured_output(dict
   schema, method="json_schema").astream`) is an assumption, verified first by the smoke
   test.
   - If Gemini/LangChain yields only the final object: stream plain text with `[n]`
     markers instead.
   - The citations would then be taken from those markers and checked against the
     known refs.
   - That is a deliberate, documented exception to "structured output, no regex", and
     needs your OK at that point.
3. **`retrieval_community_min_similarity = 0.5`** is a guess; calibrate it from the
   golden-set run.
4. **Langfuse Cloud region (EU/US) and plan quota.** The default base URL is EU; set
   `LANGFUSE_BASE_URL` if your project is in the US region.
