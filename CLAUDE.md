# ai-financial-research-assistant

GraphRAG research assistant for financial documents. Pipeline: document ingestion → chunking → LLM entity/relationship extraction → knowledge graph in PostgreSQL + pgvector → Louvain community detection → hybrid retrieval (BM25 + dense) → LangGraph agent answering questions.

<!--
Notatki dla Filipa (komentarze HTML są usuwane, zanim plik trafi do Claude, więc nie zużywają kontekstu):
- Trzymaj ten plik poniżej ~150 linii. Dopisuj regułę dopiero wtedy, gdy Claude drugi raz popełni ten sam błąd.
- Reguły dla konkretnych katalogów są w .claude/rules/ (ładują się tylko przy pracy z pasującymi plikami).
- Procedury (checklisty, workflowy) trzymaj w .claude/skills/, nie tutaj.
-->

## Environment
- Postgres + pgvector: compose service `db` (ParadeDB pg17, which provides pgvector and BM25), host port 5433 → container 5432. The app builds the URL from `POSTGRES_*` in `.env` (or `DATABASE_URL_OVERRIDE`), see `database_url` in `financial_assistant/core/config.py`. RabbitMQ (`rabbitmq` service) is configured the same way via `RABBITMQ_*`.
- When adding a new setting, update `.env.example` and the settings model together.

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
- Run the API: `uv run uvicorn financial_assistant.api.server:app --reload --host 0.0.0.0 --port 8000`
- Run ingestion / graph build: `docker compose up -d` starts the `migrate`, `broker-init`, `worker`, `outbox`, `failure-reconciler` and `storage-cleanup` services; ingestion is triggered by uploading a PDF through the API.
- Admin scripts (orphaned documents, purging deleted users): see "Admin scripts" in `README.md`.
- Migrations: `uv run alembic revision --autogenerate -m "<msg>"`, `uv run alembic upgrade head`
- Test tiers (`unit`, `integration`, `e2e`, `llm`), markers, fixtures and env gates are described in `.claude/rules/tests.md`.
- DB tests never touch the dev database: `tests/conftest.py` points `DATABASE_URL_OVERRIDE` at `<POSTGRES_DB>_test` on the same server and migrates it once per session. If that migration fails after a branch switch (the test DB is at a revision missing from the branch), ask user to run `DROP DATABASE "<POSTGRES_DB>_test" WITH (FORCE)` and rerun pytest.

## Architecture
- `financial_assistant/api`: FastAPI app (`server.py`), routers for auth and documents. `core`: settings, DB session/pools, Celery app, RabbitMQ messaging, outbox, document storage, auth (`security.py`: password hashing, JWT). `models`/`schemas`: SQLAlchemy and Pydantic. `services`: `python -m` process entrypoints run by compose (`configure_broker`, `publish_outbox`, `reconcile_ingestion`, `cleanup_storage`) and admin scripts (`publish_orphaned_documents`, `purge_deleted_users`). `alembic/`: migrations.
- Upload flow is transactional-outbox based: the API stores the PDF and a `document_outbox` row, `services/publish_outbox.py` publishes it to RabbitMQ, and the Celery task in `tasks/document_ingestion.py` runs `ai/document_ingestion.py` (acks late, one document locked with `FOR NO KEY UPDATE NOWAIT`). `services/reconcile_ingestion.py` and `services/cleanup_storage.py` are loop services.
- Account deletion (`core/accounts.py`): `owner_id IS NULL` means public, so a user who still owns documents is tombstoned (anonymized, inactive, `user.deleted_at` set) instead of deleted; `User.documents` uses `passive_deletes="all"`, so deleting such a user raises `IntegrityError` rather than publishing the documents.
- `ai/`: `document_ingestion` (PDF → parent/child chunks → embeddings), `graph_extraction` (LLM entities/relationships), `community_detection` (networkx, Louvain, plus LLM community summaries), `checkpoints` (reuse AI results of failed attempts), `requests` (timeouts/retries for model calls).
- Models: Gemini via `langchain-google-genai`; embeddings `models/gemini-embedding-001` at 768 dims (`Vector(768)` in `models/document.py` and `models/graph.py`).
- Not implemented yet: hybrid retrieval, the LangGraph agent and its API. `child_chunk` has a BM25 index (migration `95900fffcf36`); `init_langgraph_pool` in `core/db.py` is the only LangGraph piece so far.

## Compact instructions
When compacting, keep: the current branch, the list of modified files, the verification commands used, and any failing tests with their error messages.
