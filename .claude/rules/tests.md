---
paths:
  - "tests/**"
  - "**/test_*.py"
  - "**/*_test.py"
  - "**/conftest.py"
---

# Testing rules (project-specific)

- Tests are organised in tiers. Put a new test in the **lowest tier that fits**:
  - `tests/unit/` (`unit`): nothing external, no Postgres, RabbitMQ or network. Fakes and mocks only.
  - `tests/integration/` (`integration`): exactly one real dependency (Postgres **or** RabbitMQ); everything else is faked (LLM, embeddings, the broker when Postgres is real). API tests through the httpx `client` fixture belong here.
  - `tests/e2e/` (`e2e`): the full pipeline, API/outbox → RabbitMQ → a real Celery worker subprocess (`tests/e2e/ingestion_worker.py`) → Postgres, with deterministic fake model responses.
  - `tests/llm/` (`llm`): calls paid LLM/embedding APIs.
- Declare the tier in every new test module with `pytestmark = pytest.mark.<tier>` (a list if the module also needs a `skipif`). Markers are what `-m` selects on; nothing checks that the marker matches the directory, so keep them consistent.
- Inside each tier, mirror the `financial_assistant` package layout (e.g. `financial_assistant/core/outbox.py` → `tests/unit/core/test_outbox.py`). Every directory under `tests/` needs an `__init__.py`, or duplicate basenames across tiers clash at import.
- Shared Postgres/API fixtures live in `tests/fixtures/database.py`. Only the `integration` and `e2e` conftests import them (each only the ones its tests need; add the import there when a test needs another one), so unit tests cannot request them. Keep the imports in `__all__` so ruff doesn't remove them. Factory protocols and `provider_error` live in `tests/utils.py`.
- Don't add `@pytest.mark.db` by hand: tests get it automatically when they use a fixture listed in `_DB_FIXTURES` in `tests/conftest.py` (`migrated_database`, `session`, `private_database`, `outbox_db`). A new fixture that connects to Postgres must depend on `migrated_database` (it creates and migrates the `<POSTGRES_DB>_test` database and refuses to run against any database not ending in `_test`) and be added to that set.
- `llm` tests are skipped unless `RUN_LLM_TESTS=1`. The RabbitMQ integration test (`tests/integration/core/test_messaging.py`) and the e2e test are additionally gated by `RUN_RABBITMQ_TESTS=1` / `RUN_INGESTION_TESTS=1`. CI sets both and runs a plain `uv run pytest`, which therefore never calls paid APIs.
