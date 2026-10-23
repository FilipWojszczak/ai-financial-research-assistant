from functools import lru_cache
from pathlib import Path
from typing import Literal, Self
from urllib.parse import quote

from pydantic import Field, SecretStr, computed_field, model_validator
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
    rabbitmq_management_port: int = Field(default=15672, ge=1, le=65535)
    # How long one ingestion message may stay unacknowledged. Tasks acknowledge only
    # when they finish (acks_late), so this must exceed the longest document's
    # ingestion. RabbitMQ's 30-minute default closes the channel of a healthy worker.
    ingestion_consumer_timeout_seconds: int = Field(default=3 * 60 * 60, gt=0)

    document_storage_path: Path = Path("data/documents")
    # Per-call deadlines. Normal calls take <20 s (LLM) and <5 s (embedding batch);
    # a call silent past its deadline is treated as stalled and sent again.
    ai_request_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
    embedding_request_timeout_seconds: float = Field(
        default=30, gt=0, allow_inf_nan=False
    )
    # Tries per call before the timeout fails the whole ingestion attempt.
    ai_request_attempts: int = Field(default=3, ge=1, le=10)
    # None follows the environment (on in development). It logs every chat question
    # and query vector, so .env.example turns it off.
    database_echo: bool | None = None

    # Langfuse tracing is enabled only when both keys are set.
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    # EU region; the US region is https://us.cloud.langfuse.com.
    langfuse_base_url: str = "https://cloud.langfuse.com"

    chat_agent_model: str = "gemini-3.5-flash-lite"
    chat_grader_model: str = "gemini-3.5-flash-lite"
    chat_answer_model: str = "gemini-3.6-flash"
    chat_answer_temperature: float = Field(default=0.2, ge=0, le=2)
    chat_answer_max_output_tokens: int = Field(default=1024, gt=0)
    # Gemini 3 thinks by default and bills thinking as output: at its default level
    # the answer model spent 730 thinking tokens on a 4-sentence answer.
    chat_answer_thinking_level: Literal["minimal", "low", "medium", "high"] = "low"
    # Agent and grader calls get the normal stall retries. The streamed answer is
    # never sent again, and the whole turn has its own deadline.
    chat_agent_timeout_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)
    chat_answer_timeout_seconds: float = Field(default=90, gt=0, allow_inf_nan=False)
    chat_turn_timeout_seconds: float = Field(default=180, gt=0, allow_inf_nan=False)
    # A `running` turn older than this was abandoned (e.g. the API crashed mid-turn).
    chat_turn_stale_seconds: float = Field(default=360, gt=0, allow_inf_nan=False)
    chat_max_retrieval_rounds: int = Field(default=2, ge=1, le=5)
    chat_max_tool_calls_per_round: int = Field(default=3, ge=1)
    # Question/answer pairs sent to the agent and answer models.
    chat_history_turns: int = Field(default=6, ge=0)
    chat_max_question_chars: int = Field(default=2000, gt=0)
    # Bounds checkpoint growth: every turn stores the whole history again.
    chat_max_turns_per_thread: int = Field(default=100, gt=0)
    # Per user and UTC day: input + output tokens across all chat models.
    chat_daily_token_budget: int = Field(default=500_000, ge=0)

    # Hybrid retrieval: BM25 and dense rankings over child chunks fused with RRF,
    # expanded to parent chunks, plus dense search over community summaries.
    retrieval_bm25_k: int = Field(default=20, gt=0)
    retrieval_dense_k: int = Field(default=20, gt=0)
    retrieval_rrf_k: int = Field(default=60, gt=0)
    retrieval_fused_children: int = Field(default=8, gt=0)
    retrieval_max_parents: int = Field(default=5, gt=0)
    # Chunk sources across all searches of a turn passed to the grader and answer.
    retrieval_max_context_sources: int = Field(default=10, gt=0)
    retrieval_community_k: int = Field(default=3, ge=0)
    # 1 - cosine distance; calibrate with the golden set.
    retrieval_community_min_similarity: float = Field(
        default=0.5, ge=-1, le=1, allow_inf_nan=False
    )
    retrieval_hnsw_ef_search: int = Field(default=100, ge=1, le=1000)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @model_validator(mode="after")
    def _check_chat_turn_stale_seconds(self) -> Self:
        if self.chat_turn_stale_seconds <= self.chat_turn_timeout_seconds:
            # A turn still inside its deadline must never be taken for abandoned.
            raise ValueError(
                "CHAT_TURN_STALE_SECONDS must exceed CHAT_TURN_TIMEOUT_SECONDS"
            )
        return self

    @property
    def database_echo_enabled(self) -> bool:
        if self.database_echo is not None:
            return self.database_echo
        return self.environment == "development"

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
    return Settings()  # pyright: ignore[reportCallIssue]
