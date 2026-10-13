# File structure cleanup

## Goal

Put modules where their role says they belong, without changing behaviour:

1. `financial_assistant/utils.py` holds only auth code (password hashing, JWT,
   `authenticate_user`), used only by the API. It moves to
   `financial_assistant/core/security.py`.
2. The four `python -m` process entrypoints at the package root (`cleanup_storage`,
   `configure_broker`, `publish_outbox`, `reconcile_ingestion`) move into a
   `financial_assistant/services/` package. Their tests move to mirror the new path.
3. `tests/unit/test_database_url.py` tests the `tests.utils.derive_test_database_url`
   helper, not `database_url` in `core/config.py`. It is renamed to
   `tests/unit/test_derive_test_database_url.py`.
4. Only `financial_assistant/models/` has an `__init__.py`; every other app directory is
   an implicit namespace package. All of them become regular packages, and Ruff's `INP`
   rule keeps it that way.

Decisions taken during the interview:

| Topic | Decision |
| --- | --- |
| New home of `utils.py` | `financial_assistant/core/security.py` |
| Package for entrypoints | `financial_assistant/services/` |
| Renamed test | `tests/unit/test_derive_test_database_url.py` |
| Lint enforcement | Enable Ruff `INP`; ignore `INP001` for `.claude/hooks/*.py` |
| Commits | One per point, in the order above |

## Non-goals

- No change to logic, signatures, test bodies or assertions; only paths and imports.
- No split of `tests/integration/test_ingestion_operations.py` into one file per module;
  it moves to `tests/integration/services/` as a whole.
- No edits to earlier specs in `docs/specs/`; they describe the state when they were
  written.
- No image rebuild: containers bind-mount the source and run with
  `--no-install-project`.

## Design

### 1. `utils.py` → `core/security.py`

- Move the file; its relative imports become `.config` and `..models`.
- Update `api/routers/auth.py`, `api/dependencies/auth.py`, `tests/fixtures/database.py`
  and `tests/e2e/test_document_ingestion.py`.
- `CLAUDE.md` Architecture: mention auth in the `core` description.

### 2. `services/` package

- Move the four modules to `financial_assistant/services/` with an empty
  `__init__.py`; relative imports gain one level (`..core`, `..models`).
- Update the `python -m` targets in `docker-compose.yml` and
  `tests/e2e/test_document_ingestion.py` to `financial_assistant.services.<name>`.
- Move `tests/unit/test_{cleanup_storage,publish_outbox,reconcile_ingestion}.py` to
  `tests/unit/services/` and `tests/integration/test_ingestion_operations.py` to
  `tests/integration/services/`, each with an `__init__.py`; update imports and the
  `_MODULE` patch target in `test_publish_outbox.py`.
- `CLAUDE.md` Architecture: list `services` and use the new paths.

### 3. Test rename

- Rename only; content unchanged.

### 4. Regular packages

- Add empty `__init__.py` to `financial_assistant/` and to `ai/`, `api/`,
  `api/dependencies/`, `api/routers/`, `core/`, `schemas/` and `tasks/`.
- `pyproject.toml`: add `INP` to `[tool.ruff.lint] select` and `INP001` to the
  `.claude/hooks/*.py` per-file ignores (a standalone script). `alembic/` is already
  excluded from Ruff.

## Edge cases

- `tests/e2e/test_document_ingestion.py` changes in commits 1 and 2; each commit stages
  only its own hunk.
- Running containers keep the old commands until recreated; `docker compose up -d`
  recreates the four services whose commands changed.

## Verification

- No leftover references outside `docs/specs/`:
  `git grep -nE 'financial_assistant\.(utils|cleanup_storage|configure_broker|publish_outbox|reconcile_ingestion)\b|from \.\.\.?utils'`
- `uv run ruff check --fix . && uv run ruff format .`, `uv run pyright`
- `uv run pytest -m "not llm and not db" -x -q`
- `uv run pytest -m integration -x -q`
- `RUN_INGESTION_TESTS=1 uv run pytest -m e2e -x -q` (starts the moved entrypoints as
  real subprocesses)
- `docker compose config -q`, `docker compose up -d`, then `docker compose ps` and the
  logs of `broker-init`, `outbox`, `failure-reconciler` and `storage-cleanup`.
