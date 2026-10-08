---
paths:
  - "alembic/**"
  - "financial_assistant/models/**"
  - "financial_assistant/core/db.py"
  - "financial_assistant/core/outbox.py"
---

Follow the `postgres-pgvector` skill.

# Database rules (project-specific)

- Migrations: `uv run alembic revision --autogenerate -m "<msg>"`, then review the generated file, then `uv run alembic upgrade head`. Indexes that are not declared in the SQLAlchemy models (like the ParadeDB BM25 index, `USING bm25`) are invisible to autogenerate: write them by hand with `op.execute`, with a matching `DROP INDEX IF EXISTS` in `downgrade` (see `95900fffcf36`). No HNSW/IVFFlat vector index exists yet.
- Naming: singular snake_case tables (`document`, `child_chunk`, `entity_relationship`); indexes `ix_<table>_<column>`, BM25 indexes `<table>_bm25_idx`; unique constraints `uix_<…>`; Postgres enums `<name>_enum`.
