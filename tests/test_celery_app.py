import pytest
from pydantic import SecretStr

from financial_assistant.core.celery_app import create_celery_app
from financial_assistant.core.config import Settings, get_settings


def test_create_celery_app_configures_safe_queue_defaults():
    """Celery should use RabbitMQ and JSON without retaining task results."""
    app = create_celery_app()

    assert app.conf.broker_url == get_settings().broker_url
    assert app.conf.task_default_queue == "document_ingestion"
    assert app.conf.task_serializer == "json"
    assert app.conf.accept_content == ["json"]
    assert app.conf.task_ignore_result is True
    assert app.conf.enable_utc is True


def test_settings_builds_broker_url_from_rabbitmq_credentials():
    """Individual RabbitMQ settings should produce an escaped AMQP URL."""
    settings = Settings(
        _env_file=None,
        secret_key="secret",
        algorithm="HS256",
        rabbitmq_default_user="worker@example.com",
        rabbitmq_default_pass=SecretStr("password with spaces"),
        rabbitmq_default_vhost="finance/reports",
        rabbitmq_host="broker",
        rabbitmq_port=5673,
    )

    assert settings.broker_url == (
        "amqp://worker%40example.com:password%20with%20spaces@"
        "broker:5673/finance%2Freports"
    )


def test_settings_rejects_missing_rabbitmq_credentials():
    """Missing broker credentials should fail instead of using unsafe defaults."""
    settings = Settings(
        _env_file=None,
        secret_key="secret",
        algorithm="HS256",
        rabbitmq_default_user=None,
        rabbitmq_default_pass=None,
        rabbitmq_default_vhost=None,
        rabbitmq_host=None,
    )

    with pytest.raises(ValueError, match="No RabbitMQ configuration"):
        _ = settings.broker_url
