from io import StringIO

from alembic import command
from alembic.config import Config


def test_fresh_install_creates_each_enum_only_once():
    """The Compose migration service must work on an empty database too."""
    sql = StringIO()
    # No config file: do not let Alembic reconfigure the test process logging.
    config = Config(output_buffer=sql)
    config.set_main_option("script_location", "alembic")
    command.upgrade(config, "head", sql=True)
    ddl = sql.getvalue()
    for name in ("document_type_enum", "document_status_enum", "entity_type_enum"):
        assert ddl.count(f"CREATE TYPE {name} AS ENUM") == 1
    assert "CREATE TABLE document_outbox" in ddl
