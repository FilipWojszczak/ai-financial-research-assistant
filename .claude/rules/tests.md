---
paths:
  - "tests/**"
  - "**/test_*.py"
  - "**/*_test.py"
  - "**/conftest.py"
---

# Testing rules (project-specific)

- Don't add `@pytest.mark.db` by hand: tests get it automatically when they use a fixture listed in `_DB_FIXTURES` in `tests/conftest.py` (`session`, `private_database`, `outbox_db`). A new fixture that connects to Postgres must be added to that set.
- `llm` tests are skipped unless `RUN_LLM_TESTS=1`. Integration tests are additionally gated by `RUN_INGESTION_TESTS=1` / `RUN_RABBITMQ_TESTS=1` (they need Postgres + RabbitMQ).
