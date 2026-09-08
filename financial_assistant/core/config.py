import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

from pydantic import SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    environment: str = "development"

    secret_key: str
    algorithm: str

    postgres_user: str | None = None
    postgres_password: SecretStr | None = None
    postgres_db: str | None = None
    postgres_host: str | None = None
    postgres_port: int | None = 5432

    database_url_override: SecretStr | None = None

    rabbitmq_user: str | None = None
    rabbitmq_password: SecretStr | None = None
    rabbitmq_vhost: str | None = None
    rabbitmq_host: str | None = None
    rabbitmq_port: int | None = 5672

    rabbitmq_url_override: SecretStr | None = None

    document_storage_path: Path = Path("data/documents")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @computed_field(repr=False)
    @property
    def database_url(self) -> str:
        if self.database_url_override:
            return self.database_url_override.get_secret_value()

        if (
            self.postgres_user
            and self.postgres_password
            and self.postgres_db
            and self.postgres_host
        ):
            user = quote(self.postgres_user, safe="")
            password = quote(self.postgres_password.get_secret_value(), safe="")
            database = quote(self.postgres_db, safe="")
            return (
                f"postgresql+psycopg://{user}:{password}@{self.postgres_host}:"
                f"{self.postgres_port}/{database}"
            )

        raise ValueError(
            "No database configuration! Set DATABASE_URL_OVERRIDE or POSTGRES_USER, "
            "POSTGRES_PASSWORD, POSTGRES_DB, and POSTGRES_HOST."
        )

    @computed_field(repr=False)
    @property
    def broker_url(self) -> str:
        if self.rabbitmq_url_override:
            return self.rabbitmq_url_override.get_secret_value()

        if (
            self.rabbitmq_user
            and self.rabbitmq_password
            and self.rabbitmq_vhost
            and self.rabbitmq_host
        ):
            user = quote(self.rabbitmq_user, safe="")
            password = quote(self.rabbitmq_password.get_secret_value(), safe="")
            vhost = quote(self.rabbitmq_vhost, safe="")
            return (
                f"amqp://{user}:{password}@{self.rabbitmq_host}:"
                f"{self.rabbitmq_port}/{vhost}"
            )

        raise ValueError(
            "No RabbitMQ configuration! Set RABBITMQ_URL_OVERRIDE or "
            "RABBITMQ_USER, RABBITMQ_PASSWORD, RABBITMQ_VHOST, "
            "and RABBITMQ_HOST."
        )


@lru_cache
def get_settings() -> Settings:
    if os.getenv("ENVIRONMENT") == "testing":
        return Settings(_env_file=".env.test")  # pyright: ignore[reportCallIssue]
    return Settings()  # pyright: ignore[reportCallIssue]
