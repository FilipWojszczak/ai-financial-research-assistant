# Postgres fixtures are imported here, not in tests/conftest.py, so unit tests can't
# request them. __all__ keeps ruff from removing the "unused" imports.
from tests.fixtures.database import private_database

__all__ = ["private_database"]
