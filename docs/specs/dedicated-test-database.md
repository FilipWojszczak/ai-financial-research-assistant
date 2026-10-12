# Dedicated test database

## Goal

Tests stop touching the dev database. Every pytest run that needs Postgres uses a
separate database on the same server, `<dev db name>_test`, created on demand and
migrated with Alembic. Per-test isolation stays as it is (transaction rollback in
`session`, a private schema in `private_database` and `outbox_db`), only now inside that
database.

At the same time, remove the `.env.test` mechanism, which never took effect: in the Dev
Container every variable it defines is already set in the process environment from
`.env` (compose `env_file`), and pydantic-settings gives environment variables priority
over dotenv files. CI doesn't create the file at all.

Decisions taken during the interview:

| Topic | Decision |
| --- | --- |
| How tests are redirected | Global: `tests/conftest.py` sets `DATABASE_URL_OVERRIDE` in `os.environ` in `pytest_configure` |
| Test DB lifecycle | Persistent; `CREATE DATABASE` if missing, `alembic upgrade head` once per pytest session |
| Test DB name | Derived: dev database name + `_test`; no new setting |
| How migrations run | `sys.executable -m alembic upgrade head` as a subprocess |
| When | Lazily, from a session-scoped fixture; unit-only runs never connect to Postgres |
| Per-test schemas | Keep `Base.metadata.create_all` in `private_database` / `outbox_db` |
| Cleanup | Remove `.env.test.example`, the `ENVIRONMENT` branch, pytest-env and every `ENVIRONMENT=testing` |
| CI | Remove the "Create vector extension" step |

## Non-goals

- No change to application settings: no `TEST_DATABASE_URL`, no new field in `Settings`,
  no change to `.env.example`.
- No change to per-test isolation strategies or to test bodies and assertions.
- No Alembic migrations per test schema; per-test schemas keep using `create_all` and
  therefore have no BM25 index. Nothing tests BM25 yet.
- No automatic drop or recreate of the test database (see the edge cases for the manual
  recovery path).
- No Testcontainers, no separate Postgres service for tests.
- No change to RabbitMQ handling (the e2e test already creates a throwaway vhost).
- No changes to `alembic/env.py` or the migrations.

## Design

### URL derivation (`tests/utils.py`)

A pure function, unit-tested:

```python
def derive_test_database_url(database_url: str) -> str:
    """Return the same URL pointing at `<database>_test`.

    A URL whose database already ends in `_test` is returned unchanged (idempotent).
    """
```

Use `sqlalchemy.engine.make_url(url).set(database=...)` and
`render_as_string(hide_password=False)`. Query parameters, the driver
(`postgresql+psycopg`), host and port are preserved. A URL without a database name raises
`ValueError`.

### Global redirect (`tests/conftest.py`)

Add a `pytest_configure(config)` hook. It runs before subdirectory conftests and test
modules are imported, so before `financial_assistant/core/db.py` builds its module-level
engine and before `financial_assistant/utils.py` calls `get_settings()` at import time.

1. Build `Settings()` directly (not the cached `get_settings()`) and read
   `database_url`. If this fails (`ValueError` from the property, or pydantic
   `ValidationError`), do nothing: unit tests must keep working without database
   configuration, and DB tests will fail in their fixtures exactly as they do today.
2. Set `os.environ["DATABASE_URL_OVERRIDE"] = derive_test_database_url(url)`.
3. Call `get_settings.cache_clear()` defensively.

Everything reads the overridden URL from then on: `core/db.py`'s engine (used by the API
under `client` and by `core/outbox.py`), `tasks/document_ingestion.py`, fixtures, and the
`alembic` subprocess, which inherits `os.environ`. The e2e workers also inherit it; the
e2e test then sets `DATABASE_URL_OVERRIDE` to its private-schema URL as before, which is
still inside the test database.

Import `Settings`/`get_settings` inside the hook, not at module top, so `tests/conftest.py`
stays light for collection.

### `migrated_database` fixture (`tests/fixtures/database.py`)

A **sync**, `scope="session"` fixture. It's named `migrated_database` rather than
`test_database` so it doesn't look like a test function:

1. Read `get_settings().database_url`. Assert that its database name ends in `_test`, and
   abort with a clear message otherwise. This is the guard against ever running DDL on
   the dev database.
2. Connect to the maintenance database (same URL, `database="postgres"`) with a sync
   SQLAlchemy engine (`postgresql+psycopg`, `isolation_level="AUTOCOMMIT"`, because
   `CREATE DATABASE` can't run inside a transaction).
3. Take `pg_advisory_lock(<fixed bigint key>)` on that connection. Advisory locks are
   scoped to the database you're connected to; since every run locks inside `postgres`,
   two concurrent pytest sessions serialize here.
4. If `SELECT 1 FROM pg_database WHERE datname = :name` returns nothing, run
   `CREATE DATABASE` with the name quoted via the dialect's identifier preparer.
5. Run `subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], check=True,
   capture_output=True, cwd=<repo root>)`. On failure, raise with Alembic's stderr in the
   message. It runs as a subprocess because `alembic/env.py` calls
   `logging.config.fileConfig()`, which disables loggers that already exist (that would
   break `caplog` tests in `test_outbox.py`). A subprocess also keeps sync Alembic out of
   the pytest-asyncio event loop.
6. Release the lock, dispose of the engine, and `yield` the test URL (or nothing).
   Nothing is torn down: the database persists across runs.

### Fixtures that change

- `session` (`tests/fixtures/database.py`): depends on `migrated_database`; remove
  `create_all`. The schema now comes from migrations, so tests run against the real
  schema (extensions, server defaults, the BM25 index).
- `private_database` (`tests/fixtures/database.py`): depends on `migrated_database`
  (which guarantees the database and its `vector`/`pg_search` extensions exist); keeps
  `CREATE SCHEMA` + `create_all`.
- `outbox_db` (`tests/integration/core/test_outbox.py`): same as `private_database`.
- `tests/integration/conftest.py` and `tests/e2e/conftest.py`: import
  `migrated_database` and add it to `__all__`. Fixtures pulled in via `session` or
  `private_database` resolve their dependencies in the importing conftest's scope.
- `tests/conftest.py`: add `migrated_database` to `_DB_FIXTURES`.

### Cleanup of the `.env.test` mechanism

- `financial_assistant/core/config.py`: `get_settings()` becomes `return Settings()`;
  drop the `os` import if it's no longer used.
- `git rm .env.test.example`. The local `.env.test` is gitignored: tell Filip to delete it
  (Claude can't read or remove `.env*` files).
- `pyproject.toml`: remove `env = ["ENVIRONMENT=testing"]`; `uv remove --dev pytest-env`.
- `tests/e2e/test_document_ingestion.py`: remove `"ENVIRONMENT": "testing"` from the
  subprocess environment.
- `.vscode/launch.json`: remove `"ENVIRONMENT": "testing"` from "Debug Tests".
- `.github/workflows/ci.yml`: remove the "Create vector extension" step.

### Documentation to update

- `CLAUDE.md`: in Commands/tests, say that DB tests use `<POSTGRES_DB>_test`, which is
  created and migrated automatically on first use; mention `migrated_database` next to
  `_DB_FIXTURES`; add the recovery command (see Edge cases).
- `.claude/rules/tests.md`: list `migrated_database` among the shared fixtures and in
  `_DB_FIXTURES`; note that a new Postgres fixture should depend on it.

## Data & migrations

- No new migration and no model change.
- New database `<POSTGRES_DB>_test` on the existing `db` service (in CI: `test_db_test`
  on the ParadeDB service container). The extensions are created by the existing
  migrations `bac3899e0118` (`vector`) and `95900fffcf36` (`pg_search`); the default
  `POSTGRES_USER` in the image is a superuser, so `CREATE DATABASE` and
  `CREATE EXTENSION` are allowed.
- The dev database is no longer touched by tests. Stale `outbox_test_*` /
  `ingestion_test_*` schemas left in the dev database by earlier crashed runs are not
  cleaned up automatically. Mention them to Filip; dropping them is a manual,
  confirmed action.

## External calls & cost

None. No LLM or embedding calls; nothing changes in what `llm` tests do.

## Edge cases

- **Postgres not running:** `pytest -m unit` and the fast tests are unaffected
  (`pytest_configure` only computes a URL; nothing connects). DB tests error in
  `migrated_database` with the connection error.
- **No database configuration** (no `POSTGRES_*`, no `DATABASE_URL_OVERRIDE`):
  `pytest_configure` leaves the environment alone; behaviour stays as it is today.
- **`DATABASE_URL_OVERRIDE` already points at `*_test`** (e.g. set by hand, or a nested
  pytest run): the derivation is idempotent, so you never get `_test_test`.
- **Guard:** `migrated_database` refuses to run unless the effective database name ends
  in `_test`. This catches a hook that didn't run, or a later regression that lets
  something reset `DATABASE_URL_OVERRIDE`.
- **Two pytest sessions at once** (Filip plus Claude): the advisory lock serializes
  create and migrate; afterwards, rollback and private schemas keep the tests isolated.
- **Diverging migrations after a branch switch** (the test database is at a revision
  that's missing from the current branch): `alembic upgrade head` fails. The fixture's
  error message names the recovery: `DROP DATABASE "<name>_test" WITH (FORCE)` (run by
  Filip; it's destructive, so Claude asks first), then rerun pytest.
- **A migration that `create_all` would have masked** (a model changed without a
  migration): `session`-based tests now fail against the migrated schema. This is
  intended; it catches missing migrations.
- **Database name that needs quoting** (hyphens, capitals): `CREATE DATABASE` uses the
  dialect's identifier quoting; `pg_database` is queried with a bound parameter.
- **Alembic's logging config:** isolated by the subprocess. Alembic's stdout and stderr
  are captured and only surfaced on failure.

## Test plan

- **Unit** (`tests/unit/test_database_url.py`, `pytestmark = pytest.mark.unit`), for
  `derive_test_database_url`:
  - appends `_test` to the database name and keeps driver, credentials (including a
    password with special characters), host, port and query parameters;
  - is idempotent for a name already ending in `_test`;
  - raises `ValueError` when the URL has no database.
- **Existing integration and e2e tests** are the regression suite for the fixture
  changes: they must pass unchanged against the test database.
- No new `db` test for `migrated_database` itself; every DB test exercises it.

## End-to-end verification steps

Run inside the Dev Container with `db` (and `rabbitmq` for the gated tests) running:

1. `uv run ruff check --fix . && uv run ruff format .` and `uv run pyright`: clean.
2. `uv run pytest -m unit -q`: passes. To show it doesn't need Postgres, run it with
   `docker compose stop db`, then start `db` again.
3. Confirm the test database doesn't exist yet:
   `psql -h db -U "$POSTGRES_USER" -d postgres -c "\l"`.
4. `uv run pytest -m integration -x -q`: passes. Then `\l` shows `<POSTGRES_DB>_test`, and
   `psql ... -d <POSTGRES_DB>_test -c "SELECT version_num FROM alembic_version"`
   shows the current head (`uv run alembic heads`).
5. `RUN_RABBITMQ_TESTS=1 RUN_INGESTION_TESTS=1 uv run pytest -x -q`: everything passes
   except `llm` (skipped).
6. Run step 4 again: no `CREATE DATABASE`, and the migration is a no-op.
7. Dev database untouched: compare row counts of `document`, `user` and
   `document_outbox` in `<POSTGRES_DB>` before and after steps 4 and 5, and
   `SELECT nspname FROM pg_namespace WHERE nspname LIKE '%_test_%'` returns no new schemas
   in it.
8. Guard: the hook always re-derives the URL, so it can't be pointed at the dev database
   from outside. Check the guard's message by temporarily calling its assertion with a
   non-`_test` URL. Never run DDL on the dev database to "prove" it.
9. `grep -rn "ENVIRONMENT\|env\.test\|pytest-env" --exclude-dir=.venv --exclude-dir=.git .`
   returns only historical hits in `docs/specs/`, and `uv pip show pytest-env` reports
   that it's not installed.
10. Push the branch and check that CI passes; the run now creates `test_db_test` and
    applies the full migration chain on a fresh database.

## Open questions

- None blocking. After merge, Filip deletes the local `.env.test` and optionally drops
  stale `*_test_*` schemas from the dev database (manual, confirmed).
