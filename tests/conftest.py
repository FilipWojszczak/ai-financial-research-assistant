import os

import pytest

# Any test that requests these (directly or through client/factories) hits Postgres.
# outbox_db lives in tests/integration/core/test_outbox.py; add new module-level DB
# fixtures here too.
_DB_FIXTURES = {"session", "private_database", "outbox_db"}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    skip_llm = pytest.mark.skip(reason="Set RUN_LLM_TESTS=1 to call paid LLM APIs")
    run_llm = os.getenv("RUN_LLM_TESTS") == "1"
    for item in items:
        if _DB_FIXTURES & set(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.db)
        if item.get_closest_marker("llm") and not run_llm:
            item.add_marker(skip_llm)
