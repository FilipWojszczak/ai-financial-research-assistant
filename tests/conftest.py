import os

import pytest

# Any test that requests these (directly or through client/factories) hits Postgres.
# outbox_db lives in tests/integration/core/test_outbox.py; add new module-level DB
# fixtures here too.
_DB_FIXTURES = {"migrated_database", "session", "private_database", "outbox_db"}


def _redirect_to_test_database() -> None:
    # Runs at import time, not in a pytest hook: conftests on the command-line paths
    # (e.g. `pytest tests/integration/...`) are imported before any hook runs and
    # import core/db.py, which builds its engine from the settings at that moment.
    # This root conftest is imported before them. Subprocesses (Alembic, e2e workers)
    # inherit the override. Nothing connects here.
    from financial_assistant.core.config import Settings, get_settings
    from tests.utils import derive_test_database_url

    try:
        url = derive_test_database_url(Settings().database_url)  # pyright: ignore[reportCallIssue]
    except ValueError:  # Includes pydantic's ValidationError.
        # No database configuration: unit tests still run; DB fixtures fail as before.
        return
    os.environ["DATABASE_URL_OVERRIDE"] = url
    get_settings.cache_clear()


_redirect_to_test_database()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    skip_llm = pytest.mark.skip(reason="Set RUN_LLM_TESTS=1 to call paid LLM APIs")
    run_llm = os.getenv("RUN_LLM_TESTS") == "1"
    for item in items:
        if _DB_FIXTURES & set(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.db)
        if item.get_closest_marker("llm") and not run_llm:
            item.add_marker(skip_llm)
