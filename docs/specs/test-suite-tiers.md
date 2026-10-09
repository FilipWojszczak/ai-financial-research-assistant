# Test suite reorganisation into tiers

## Goal

Reorganise `tests/` into explicit tiers so that it is obvious, from both the path and
the markers, what a test needs to run and how expensive it is:

| Tier          | Directory            | Marker                | Needs                                                                 |
|---------------|----------------------|-----------------------|-----------------------------------------------------------------------|
| unit          | `tests/unit/`        | `unit`                | Nothing external: no Postgres, no RabbitMQ, no network. Fakes/mocks only. |
| integration   | `tests/integration/` | `integration`         | Exactly one real dependency (Postgres **or** RabbitMQ); everything else faked (LLM, embeddings, broker when Postgres is real). API tests via the httpx `client` live here. |
| e2e           | `tests/e2e/`         | `e2e`                 | Full pipeline: API/outbox → RabbitMQ → real Celery worker subprocess → Postgres, with deterministic fake model responses. |
| llm           | `tests/llm/`         | `llm` (existing)      | Paid LLM/embedding APIs. Empty for now; reserved for future extraction/retrieval quality tests. |

Inside each tier, files mirror the `financial_assistant` package layout.

Tier markers are the **primary** selection mechanism (explicit `pytestmark` per module);
directories are for organisation and browsing. No collection-time guard enforces that
marker and directory agree (decided: rely on convention and review).

## Non-goals

- No changes to test bodies or assertions beyond what moving requires (imports,
  fixture locations, the worker module path).
- No new tests, no coverage changes.
- No change to env-var gating: `RUN_INGESTION_TESTS=1` and `RUN_RABBITMQ_TESTS=1` stay as
  they are; no new `RUN_E2E_TESTS` gate.
- No change to `.github/workflows/ci.yml`: CI keeps a single `uv run pytest` step.
- No collection-time check that tier markers match directories.
- No changes to `financial_assistant/` application code.

## Design

### Directory layout and move map

Every directory gets an empty `__init__.py` so that same basenames in different tiers
(e.g. `unit/core/test_outbox.py` and `integration/core/test_outbox.py`) don't clash under
pytest's default `prepend` import mode, and so `from tests.utils import ...` keeps working.

Use `git mv` for whole-file moves so history follows. For split files, `git mv` the file
to the tier holding the majority of its tests, then cut the remaining tests into the new
file.

Counts are test functions (parametrised cases count once); the suite currently collects
**167 items / 123 functions**, and both numbers must be unchanged afterwards.

**unit (70 functions)**

| From | To | Functions |
|------|----|-----------|
| `test_ai_requests.py` | `unit/ai/test_requests.py` | 8 |
| `test_community_detection.py` | `unit/ai/test_community_detection.py` | 10 |
| `test_document_ingestion.py` | `unit/ai/test_document_ingestion.py` | 7 |
| `test_graph_extraction.py` | `unit/ai/test_graph_extraction.py` | 6 |
| `test_documents.py` (2 non-DB tests: `test_upload_document_rejects_upload_file_without_filename`, `test_upload_failure_cleans_up_only_before_commit`) | `unit/api/routers/test_documents.py` | 2 |
| `test_document_storage.py` | `unit/core/test_document_storage.py` | 5 |
| `test_messaging.py` | `unit/core/test_messaging.py` | 3 |
| `test_outbox.py::test_invalid_limit_is_rejected` | `unit/core/test_outbox.py` | 1 |
| `test_ingestion_tasks.py` | `unit/tasks/test_document_ingestion.py` | 17 |
| `test_outbox_service.py` | `unit/test_publish_outbox.py` | 7 |
| `test_ingestion_operations.py::test_failure_message_acknowledged_only_after_successful_reconciliation` | `unit/test_reconcile_ingestion.py` | 1 |
| `test_migration_startup.py` (offline `alembic upgrade --sql`, no DB) | `unit/test_migrations.py` | 1 |

Note: `test_community_detection.py` and `test_graph_extraction.py` use a local mock named
`session`, not the `session` fixture; pytest already classifies them as non-DB. Don't
name a directory `tests/unit/alembic/`, to avoid any confusion with the `alembic` package.

**integration (53 functions)**

| From | To | Functions | Real dependency |
|------|----|-----------|-----------------|
| `test_auth.py` | `integration/api/routers/test_auth.py` | 9 | Postgres |
| `test_documents.py` (remaining DB tests) | `integration/api/routers/test_documents.py` | 17 | Postgres |
| `test_outbox.py` (remaining, incl. `outbox_db` and `send` fixtures) | `integration/core/test_outbox.py` | 12 | Postgres |
| `test_rabbitmq_routing.py` | `integration/core/test_messaging.py` | 2 | RabbitMQ |
| `test_ingestion_transactions.py` | `integration/ai/test_document_ingestion.py` | 8 | Postgres |
| `test_ingestion_operations.py` (remaining DB tests) | `integration/test_ingestion_operations.py` | 5 | Postgres |

`test_ingestion_operations.py` keeps its name (exception to the mirror rule): it covers
both `reconcile_ingestion` and `cleanup_storage` through one shared `operations` fixture,
and splitting it would just move that fixture around.

**e2e (2 functions)**

| From | To |
|------|----|
| `test_ingestion_integration.py` | `e2e/test_document_ingestion.py` |
| `ingestion_worker.py` (helper, not a test) | `e2e/ingestion_worker.py` |

The worker is started as a subprocess by module path: update
`start("tests.ingestion_worker", ...)` to `start("tests.e2e.ingestion_worker", ...)`.

**llm**: create `tests/llm/__init__.py` only.

**Unchanged location**: `tests/utils.py` (factory protocols, `provider_error`), shared by tiers.

### Markers

`pyproject.toml` `[tool.pytest.ini_options]`:

```toml
addopts = "--strict-markers"
markers = [
  "unit: no external dependencies (fakes only)",
  "integration: one real dependency (Postgres or RabbitMQ), everything else faked",
  "e2e: full pipeline with a real Celery worker, RabbitMQ and Postgres",
  "llm: calls a real LLM/embedding API (costs money, slow)",
  "db: requires a running Postgres + pgvector",
]
```

Each test module declares its tier explicitly at module level:

```python
pytestmark = pytest.mark.unit
```

Modules that already have a `pytestmark` (the env-var `skipif` in the e2e and RabbitMQ
files) become a list: `pytestmark = [pytest.mark.e2e, pytest.mark.skipif(...)]`.

`db` stays auto-applied from `_DB_FIXTURES` (never hand-written), and `llm` stays
explicit with its `RUN_LLM_TESTS=1` skip. A future `tests/llm/` module will use
`pytestmark = pytest.mark.llm` as its tier marker.

### Fixtures (layered conftests)

- `tests/conftest.py`: keeps **only** `pytest_collection_modifyitems` (auto `db` marker,
  `llm` skip) and `_DB_FIXTURES`. Update the comment that says `outbox_db` lives in
  `test_outbox.py` to its new path.
- `tests/fixtures/__init__.py` + `tests/fixtures/database.py`: the current Postgres/API
  fixtures moved verbatim: `private_database`, `session`, `client`, `user_factory`,
  `token_factory`, `document_factory`.
- `tests/integration/conftest.py` and `tests/e2e/conftest.py`: import those fixtures
  from `tests.fixtures.database` so they are available only in tiers that may touch
  Postgres. Ruff will flag them as unused imports (F401). Silence that with an `__all__`
  list or `# noqa: F401` on the import, and don't let `ruff check --fix` delete them.
  `pytest_plugins` is **not** used: it is only allowed in the root conftest and would make
  the fixtures global again.
- `tests/unit/conftest.py`: not needed. Because no unit-tier conftest provides `session`
  etc., a unit test that requests a DB fixture fails with "fixture not found". That is
  the intended guard against DB use in unit tests.
- File-local fixtures (`attempt`, `close_task_loop`, `send`, `outbox_db`,
  `committed_document`, `fake_pipeline`, `graph_pipeline_with_fake_models`, `operations`,
  `flow`, `broker_queues`, `no_retry_pause`) move with the tests that use them. When a file
  is split, copy a fixture into both halves only if both use it (today only `send` in
  `test_outbox.py` might need checking; `test_invalid_limit_is_rejected` uses no fixture).

### Documentation to update

- `CLAUDE.md` → *Commands*: add per-tier commands, keeping the existing fast-test line:
  - Unit: `uv run pytest -m unit -q`
  - Fast tests (no network, no DB): `uv run pytest -m "not llm and not db" -x -q` (unchanged;
    it still selects unit tests plus env-gated tests that skip without services)
  - Integration: `uv run pytest -m integration -x -q`
  - E2E: `RUN_INGESTION_TESTS=1 uv run pytest -m e2e -x -q`
  - Describe the tier layout in one short paragraph and update the `_DB_FIXTURES` sentence
    (fixtures now live in `tests/fixtures/database.py`).
- `.claude/rules/tests.md`: tier definitions, "declare `pytestmark = pytest.mark.<tier>` in
  every new test module", "put the file in the lowest tier that fits", the mirror-layout
  rule, and the shared-fixture location.
- `.claude/agents/code-reviewer.md`: the `_DB_FIXTURES in tests/conftest.py` reference stays
  valid; add "new test module without a tier marker" to its checklist.

## Data & migrations

None. No schema, model or migration changes. `test_migrations.py` only moves.

## External calls & cost

None. No test gains or loses an LLM/embedding call. There are no `llm`-marked tests today,
so a plain `uv run pytest` still never calls paid APIs.

## Edge cases

- **Import-mode collisions**: missing `__init__.py` in any new directory produces
  "import file mismatch" for duplicate basenames. Every directory under `tests/` (including
  `fixtures/`, `llm/` and the nested `ai/`, `api/`, `api/routers/`, `core/`, `tasks/`) needs one.
- **Subprocess worker path**: the e2e test starts Celery with `-A`/module
  `tests.ingestion_worker`. If it isn't updated, the worker fails to import and the e2e test
  hangs until its timeout instead of failing fast. Grep for `ingestion_worker` after the move.
- **Fixture visibility**: a fixture used by an e2e test but only imported in the
  integration conftest yields "fixture not found". The e2e test uses `client`,
  `private_database` and `session`, so all three must be imported in `tests/e2e/conftest.py`.
- **Ruff autofix**: the stop-quality-gate hook and `ruff check --fix` would delete
  fixture re-exports flagged F401. Use `__all__` or `noqa` (see above).
- **`--strict-markers`**: any existing unregistered marker would now error at
  collection. Today only `parametrize`, `skipif`, `llm` and `db` are used, so this should be
  clean. Verify with `--collect-only`.
- **Marker drift**: with no guard, a file moved to another tier but keeping its old
  `pytestmark` is selected by the wrong `-m`. The verification step below checks this
  once; afterwards it relies on review.
- **`_DB_FIXTURES` auto-marking** keys off fixture names, so it keeps working regardless
  of where the fixtures are defined.

## Test plan

This change is a pure refactor of the test suite, so the tests are their own oracle.
Success means the same tests are collected and pass, now carrying the right markers.

1. Before any change, save a baseline:
   `uv run pytest --collect-only -q | grep '::' | sed 's|.*::||' | sort > <scratchpad>/before.txt`
   (test node names without paths), and record the fast-test and DB-test results.
2. After the change, produce `after.txt` the same way. `diff` must be empty: same 167
   items, nothing lost or duplicated by the splits.
3. Per-tier collection counts must add up: `-m unit` + `-m integration` + `-m e2e` = 167
   and `-m "not unit and not integration and not e2e"` collects 0.
4. `-m unit` must not include any test also marked `db`:
   `uv run pytest --collect-only -q -m "unit and db"` collects 0.
5. Run the suites as before:
   - `uv run pytest -m "not llm and not db" -x -q`: same pass count as the baseline.
   - `uv run pytest -m db -x -q` (Postgres up): same pass count as the baseline.

## End-to-end verification steps

1. `uv run ruff check --fix . && uv run ruff format .`: clean. Confirm that fixture
   imports in tier conftests survived the autofix.
2. `uv run pyright`: no new errors.
3. Collection diff and per-tier counts (Test plan steps 1–4), with commands and output
   shown.
4. `uv run pytest -m unit -q`: passes with Postgres and RabbitMQ **stopped**. This proves
   that the unit tier is really dependency-free.
5. With `docker compose up -d db rabbitmq`:
   `RUN_INGESTION_TESTS=1 RUN_RABBITMQ_TESTS=1 uv run pytest -q`: whole suite green,
   same totals as the baseline. This is what CI runs.
6. `git log --follow tests/unit/ai/test_requests.py` shows the pre-move history (rename
   detection worked).
7. Push the branch (`chore/test-suite-tiers`) and confirm that the unchanged CI job passes.

## Open questions

- Should CI later be split into per-tier jobs (unit without services, then integration +
  e2e)? It was decided against for now. Revisit when the suite gets slow.
- Should the two env gates be merged into a single `RUN_E2E_TESTS` gate? Not now. Note that
  `RUN_RABBITMQ_TESTS` then gates an *integration* test and `RUN_INGESTION_TESTS` gates the
  *e2e* test, so the names no longer describe the tiers.
- Is a marker/directory drift guard worth adding once more people contribute? Not now.
