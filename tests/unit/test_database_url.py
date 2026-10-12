import pytest
from sqlalchemy.engine import make_url

from tests.utils import derive_test_database_url

pytestmark = pytest.mark.unit


def test_appends_suffix_and_keeps_other_url_parts():
    url = make_url("postgresql+psycopg://app:x@db:5433/finance?sslmode=disable").set(
        password="p@ss:w/rd%#"
    )

    derived = make_url(derive_test_database_url(url.render_as_string(False)))

    assert derived.database == "finance_test"
    assert derived.drivername == "postgresql+psycopg"
    assert derived.username == "app"
    assert derived.password == "p@ss:w/rd%#"
    assert (derived.host, derived.port) == ("db", 5433)
    assert derived.query == {"sslmode": "disable"}


def test_is_idempotent():
    url = derive_test_database_url("postgresql+psycopg://app:secret@db:5432/finance")

    assert derive_test_database_url(url) == url
    assert make_url(url).database == "finance_test"


def test_rejects_url_without_database():
    with pytest.raises(ValueError, match="no database name"):
        derive_test_database_url("postgresql+psycopg://app:secret@db:5432")
