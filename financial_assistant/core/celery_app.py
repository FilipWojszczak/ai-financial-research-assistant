from celery import Celery

from .config import get_settings


def create_celery_app() -> Celery:
    """Create the Celery application shared by task producers and workers."""
    settings = get_settings()
    app = Celery("financial_assistant", broker=settings.broker_url)

    # JSON prevents Celery from deserializing arbitrary Python objects.
    app.conf.update(
        accept_content=["json"],
        broker_connection_retry_on_startup=True,
        enable_utc=True,
        result_serializer="json",
        task_default_queue="document_ingestion",
        task_ignore_result=True,
        task_serializer="json",
        timezone="UTC",
    )
    return app


celery_app = create_celery_app()
