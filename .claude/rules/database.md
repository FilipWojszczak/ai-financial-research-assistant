---
paths:
  - "alembic/**"
  - "financial_assistant/models/**"
  - "financial_assistant/core/db.py"
  - "financial_assistant/core/outbox.py"
---

# Database / pgvector rules

- Schema changes only through Alembic migrations: `uv run alembic revision --autogenerate -m "<msg>"` (review the generated file, it must include pgvector/BM25 index details), then `uv run alembic upgrade head`. Never edit a migration that has already been applied.
- pgvector: the distance operator in a query must match the index opclass — `<=>` ↔ `vector_cosine_ops`, `<->` ↔ `vector_l2_ops`, `<#>` ↔ `vector_ip_ops`. Otherwise the index is silently not used.
- When changing a retrieval query, check the plan with `EXPLAIN ANALYZE` and report whether the HNSW/IVFFlat index is used.
- Apply metadata filters in a way that does not break the vector index (check the plan); watch for top-k returning fewer rows than expected after filtering.
- Insert embeddings, chunks and graph edges in batches, never row by row in a loop.
- With async SQLAlchemy: one session per request/task, never share a session between concurrent tasks, no sync DB calls inside async code.
- Never run DROP / TRUNCATE / DELETE without WHERE without asking first.
