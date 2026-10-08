"""Declare the ingestion and dead-letter queues before starting workers."""

import base64
import json
import logging
import urllib.request
from contextlib import closing
from urllib.parse import quote

from kombu import Connection

from .core.celery_app import celery_app
from .core.config import get_settings
from .core.messaging import (
    INGESTION_POLICY_NAME,
    declare_ingestion_topology,
    ingestion_policy,
)

logger = logging.getLogger(__name__)
_MANAGEMENT_TIMEOUT_SECONDS = 5


def policy_request(broker_url: str, management_port: int, consumer_timeout: int):
    """Build the management API request that sets the ingestion queue policy."""
    broker = Connection(broker_url)
    hostname = broker.hostname
    if hostname is None:
        raise ValueError("The broker URL has no hostname")
    host = f"[{hostname}]" if ":" in hostname else hostname
    scheme = "https" if broker.ssl else "http"
    vhost = quote(broker.virtual_host or "/", safe="")
    url = (
        f"{scheme}://{host}:{management_port}"
        f"/api/policies/{vhost}/{quote(INGESTION_POLICY_NAME, safe='')}"
    )
    credentials = f"{broker.userid}:{broker.password}".encode()
    return urllib.request.Request(  # noqa: S310 - scheme is http(s) only
        url,
        data=json.dumps(ingestion_policy(consumer_timeout)).encode(),
        method="PUT",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Basic " + base64.b64encode(credentials).decode(),
        },
    )


def apply_ingestion_policy() -> None:
    """Idempotently create or update the policy through the management API."""
    settings = get_settings()
    request = policy_request(
        settings.broker_url,
        settings.rabbitmq_management_port,
        settings.ingestion_consumer_timeout_seconds,
    )
    # Raises HTTPError for 4xx/5xx, e.g. 401 when the user lacks the policymaker tag.
    with urllib.request.urlopen(  # noqa: S310 - URL built from broker settings
        request, timeout=_MANAGEMENT_TIMEOUT_SECONDS
    ):
        pass
    logger.info(
        "Applied RabbitMQ policy %s (consumer-timeout %d s)",
        INGESTION_POLICY_NAME,
        settings.ingestion_consumer_timeout_seconds,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    with celery_app.connection_for_write(
        connect_timeout=5,
        transport_options={"read_timeout": 5, "write_timeout": 5},
    ) as connection:
        connection.ensure_connection(max_retries=3)
        with closing(connection.channel()) as channel:
            declare_ingestion_topology(channel)
    apply_ingestion_policy()


if __name__ == "__main__":
    main()
