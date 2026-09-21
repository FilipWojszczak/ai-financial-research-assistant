from celery import Celery

from .config import get_settings
from .messaging import (
    INGESTION_EXCHANGE,
    INGESTION_QUEUE,
    INGESTION_ROUTING_KEY,
    ingestion_queue,
)


def create_celery_app() -> Celery:
    """Create the Celery application shared by task producers and workers."""
    settings = get_settings()
    app = Celery(
        "financial_assistant",
        broker=settings.broker_url,
        include=["financial_assistant.tasks.document_ingestion"],
    )

    # JSON prevents Celery from deserializing arbitrary Python objects.
    app.conf.update(
        accept_content=["json"],
        broker_connection_retry_on_startup=True,
        broker_transport_options={"confirm_publish": True},
        enable_utc=True,
        task_default_queue=INGESTION_QUEUE,
        task_default_exchange=INGESTION_EXCHANGE,
        task_default_exchange_type="topic",
        task_default_routing_key=INGESTION_ROUTING_KEY,
        task_queues=(ingestion_queue,),
        task_create_missing_queues=False,
        worker_detect_quorum_queues=True,
        worker_cancel_long_running_tasks_on_connection_loss=True,
        task_default_delivery_mode="persistent",
        task_ignore_result=True,
        task_serializer="json",
        timezone="UTC",
        worker_prefetch_multiplier=1,
    )
    return app


celery_app = create_celery_app()
