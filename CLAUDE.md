# ai-financial-research-assistant

GraphRAG research assistant for financial documents. Pipeline: document ingestion → chunking → LLM entity/relationship extraction → knowledge graph in PostgreSQL + pgvector → Louvain community detection → hybrid retrieval (BM25 + dense) → LangGraph agent answering questions.

<!--
Notatki dla Filipa (komentarze HTML są usuwane, zanim plik trafi do Claude, więc nie zużywają kontekstu):
- Trzymaj ten plik poniżej ~150 linii. Dopisuj regułę dopiero wtedy, gdy Claude drugi raz popełni ten sam błąd.
- Reguły dla konkretnych katalogów są w .claude/rules/ (ładują się tylko przy pracy z pasującymi plikami).
- Procedury (checklisty, workflowy) trzymaj w .claude/skills/, nie tutaj.
-->

## Environment
- All code runs inside the Dev Container. Never install anything on the host.
- Python dependencies are managed only with uv: `uv add <pkg>`, `uv add --dev <pkg>`, `uv remove <pkg>`. Never use pip, never edit `uv.lock` by hand.
- Postgres + pgvector: compose service `db` (ParadeDB pg17, which provides pgvector and BM25), host port 5433 → container 5432. The app builds the URL from `POSTGRES_*` in `.env` (or `DATABASE_URL_OVERRIDE`), see `database_url` in `financial_assistant/core/config.py`. RabbitMQ (`rabbitmq` service) is configured the same way via `RABBITMQ_*`.
- Secrets live in `.env`, which you cannot read. Variable names are documented in `.env.example`; when adding a new setting, update `.env.example` and the settings model together.

## Commands
- Sync dependencies: `uv sync`
- Unit tests: `uv run pytest -m unit -q`
- Fast tests (no network, no DB): `uv run pytest -m "not llm and not db" -x -q` (unit tests plus env-gated tests that skip without services)
- Integration tests (Postgres must be running; also RabbitMQ with `RUN_RABBITMQ_TESTS=1`): `uv run pytest -m integration -x -q`
- E2E tests (Postgres + RabbitMQ): `RUN_INGESTION_TESTS=1 uv run pytest -m e2e -x -q`
- Single test: `uv run pytest tests/path/test_file.py::test_name -x -q`
- DB tests (Postgres must be running): `uv run pytest -m db -x -q`
- Lint & format: `uv run ruff check --fix . && uv run ruff format .`
- Type check: `uv run pyright`
- Installed version of a library: `uv pip show <package>`
- Run the API: `uv run uvicorn financial_assistant.api.server:app --reload --host 0.0.0.0 --port 8000`
- Run ingestion / graph build: `docker compose up -d` starts the `migrate`, `broker-init`, `worker`, `outbox`, `failure-reconciler` and `storage-cleanup` services; ingestion is triggered by uploading a PDF through the API.
- Migrations: `uv run alembic revision --autogenerate -m "<msg>"`, `uv run alembic upgrade head`
- Tests are split into tiers by directory and marker: `tests/unit/` (`unit`, no external dependencies), `tests/integration/` (`integration`, exactly one real dependency: Postgres or RabbitMQ), `tests/e2e/` (`e2e`, API → RabbitMQ → real Celery worker → Postgres with fake models) and `tests/llm/` (`llm`, paid APIs). Inside each tier, files mirror the `financial_assistant` layout. Every test module declares `pytestmark = pytest.mark.<tier>`; markers are the selection mechanism.
- `tests/conftest.py` marks every test that uses a fixture in `_DB_FIXTURES` (`session`, `private_database`, `outbox_db`; directly or via `client`/factories) as `db` automatically. A new fixture that connects to Postgres must be added to that set. Shared Postgres/API fixtures live in `tests/fixtures/database.py` and are imported only by the `integration` and `e2e` conftests. `llm` tests are skipped unless `RUN_LLM_TESTS=1`, so a plain `uv run pytest` (what CI runs) never calls paid APIs.
- `RUN_RABBITMQ_TESTS=1` gates the RabbitMQ integration test (`tests/integration/core/test_messaging.py`) and `RUN_INGESTION_TESTS=1` gates the e2e test (both need Postgres + RabbitMQ); CI sets both.

## Architecture
- `financial_assistant/api`: FastAPI app (`server.py`), routers for auth and documents. `core`: settings, DB session/pools, Celery app, RabbitMQ messaging, outbox, document storage. `models`/`schemas`: SQLAlchemy and Pydantic. `alembic/`: migrations.
- Upload flow is transactional-outbox based: the API stores the PDF and a `document_outbox` row, `publish_outbox` publishes it to RabbitMQ, and the Celery task in `tasks/document_ingestion.py` runs `ai/document_ingestion.py` (acks late, one document locked with `FOR NO KEY UPDATE NOWAIT`). `reconcile_ingestion` and `cleanup_storage` are loop services.
- `ai/`: `document_ingestion` (PDF → parent/child chunks → embeddings), `graph_extraction` (LLM entities/relationships), `community_detection` (networkx, Louvain, plus LLM community summaries), `checkpoints` (reuse AI results of failed attempts), `requests` (timeouts/retries for model calls).
- Models: Gemini via `langchain-google-genai`; embeddings `models/gemini-embedding-001` at 768 dims (`Vector(768)` in `models/document.py` and `models/graph.py`).
- Not implemented yet: hybrid retrieval, the LangGraph agent and its API. `child_chunk` has a BM25 index (migration `95900fffcf36`); `init_langgraph_pool` in `core/db.py` is the only LangGraph piece so far.

## Rules
- The embedding dimension must match the pgvector column. Changing the embedding model or the chunking strategy requires a migration and re-embedding: flag it and ask, never do it silently.

## Compact instructions
When compacting, keep: the current branch, the list of modified files, the verification commands used, and any failing tests with their error messages.
