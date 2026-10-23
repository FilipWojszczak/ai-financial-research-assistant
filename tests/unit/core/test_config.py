import pytest
from pydantic import ValidationError

from financial_assistant.core.config import Settings

pytestmark = pytest.mark.unit


def _settings(**overrides) -> Settings:
    return Settings(
        secret_key="test",
        algorithm="HS256",
        _env_file=None,  # pyright: ignore[reportCallIssue]
        **overrides,
    )


@pytest.mark.parametrize(
    ("environment", "database_echo", "expected"),
    [
        ("development", None, True),
        ("production", None, False),
        ("development", False, False),
        ("production", True, True),
    ],
)
def test_database_echo_follows_the_environment_unless_set(
    environment, database_echo, expected
):
    settings = _settings(environment=environment, database_echo=database_echo)
    assert settings.database_echo_enabled is expected


def test_stale_turn_threshold_must_exceed_the_turn_timeout():
    with pytest.raises(ValidationError, match="CHAT_TURN_STALE_SECONDS"):
        _settings(chat_turn_timeout_seconds=180, chat_turn_stale_seconds=180)
    assert _settings(chat_turn_timeout_seconds=180, chat_turn_stale_seconds=181)
