from celery import Celery

from .config import get_settings


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
        task_default_queue="document_ingestion",
        task_default_delivery_mode="persistent",
        task_ignore_result=True,
        task_serializer="json",
        timezone="UTC",
        worker_prefetch_multiplier=1,
    )
    return app


celery_app = create_celery_app()
