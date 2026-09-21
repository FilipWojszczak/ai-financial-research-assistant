"""Declare the ingestion and dead-letter queues before starting workers."""

from .core.celery_app import celery_app
from .core.messaging import declare_ingestion_topology


def main() -> None:
    with celery_app.connection_for_write(
        connect_timeout=5,
        transport_options={"read_timeout": 5, "write_timeout": 5},
    ) as connection:
        connection.ensure_connection(max_retries=3)
        with connection.channel() as channel:
            declare_ingestion_topology(channel)


if __name__ == "__main__":
    main()
