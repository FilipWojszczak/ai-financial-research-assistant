import os
from functools import lru_cache
from urllib.parse import quote

from pydantic import SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    environment: str = "development"

    secret_key: str
    algorithm: str

    postgres_user: str | None = None
    postgres_password: str | None = None
    postgres_db: str | None = None
    postgres_host: str | None = None
    postgres_port: int | None = 5432

    _database_url: str | None = None

    rabbitmq_default_user: str | None = None
    rabbitmq_default_pass: SecretStr | None = None
    rabbitmq_default_vhost: str | None = None
    rabbitmq_host: str | None = None
    rabbitmq_port: int | None = 5672

    # Managed RabbitMQ providers commonly supply one complete AMQP URL. When it is
    # absent, build the URL from the individual settings used by Docker Compose.
    celery_broker_url: SecretStr | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @computed_field
    @property
    def database_url(self) -> str:
        if self._database_url:
            return self._database_url
        if all(
            [
                self.postgres_user,
                self.postgres_password,
                self.postgres_db,
                self.postgres_host,
            ]
        ):
            return (
                f"postgresql+psycopg://{self.postgres_user}:"
                f"{self.postgres_password}@{self.postgres_host}:"
                f"{self.postgres_port}/{self.postgres_db}"
            )
        raise ValueError(
            "No database configuration! Set _DATABASE_URL or POSTGRES_* variable set."
        )

    @computed_field(repr=False)
    @property
    def broker_url(self) -> str:
        if self.celery_broker_url:
            return self.celery_broker_url.get_secret_value()

        if (
            self.rabbitmq_default_user
            and self.rabbitmq_default_pass
            and self.rabbitmq_default_vhost
            and self.rabbitmq_host
        ):
            user = quote(self.rabbitmq_default_user, safe="")
            password = quote(self.rabbitmq_default_pass.get_secret_value(), safe="")
            vhost = quote(self.rabbitmq_default_vhost, safe="")
            return (
                f"amqp://{user}:{password}@{self.rabbitmq_host}:"
                f"{self.rabbitmq_port}/{vhost}"
            )

        raise ValueError(
            "No RabbitMQ configuration! Set CELERY_BROKER_URL or "
            "RABBITMQ_DEFAULT_USER, RABBITMQ_DEFAULT_PASS, "
            "RABBITMQ_DEFAULT_VHOST, and RABBITMQ_HOST."
        )


@lru_cache
def get_settings() -> Settings:
    if os.getenv("ENVIRONMENT") == "testing":
        return Settings(_env_file=".env.test")  # pyright: ignore[reportCallIssue]
    return Settings()  # pyright: ignore[reportCallIssue]
