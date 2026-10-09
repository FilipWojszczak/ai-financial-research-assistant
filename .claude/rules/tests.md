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
- Shared Postgres/API fixtures (`private_database`, `session`, `client`, `user_factory`, `token_factory`, `document_factory`) live in `tests/fixtures/database.py` and are imported only by the `integration` and `e2e` conftests: `tests/integration/conftest.py` imports all six, `tests/e2e/conftest.py` only `private_database` (add imports there when an e2e test needs more). Imports are kept in `__all__` so ruff doesn't remove them. Unit tests therefore cannot request them. Factory protocols and `provider_error` live in `tests/utils.py`.
- Don't add `@pytest.mark.db` by hand: tests get it automatically when they use a fixture listed in `_DB_FIXTURES` in `tests/conftest.py` (`session`, `private_database`, `outbox_db`). A new fixture that connects to Postgres must be added to that set.
- `llm` tests are skipped unless `RUN_LLM_TESTS=1`. The RabbitMQ integration test and the e2e test are additionally gated by `RUN_RABBITMQ_TESTS=1` / `RUN_INGESTION_TESTS=1`.
