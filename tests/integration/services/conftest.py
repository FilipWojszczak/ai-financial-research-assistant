from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from financial_assistant.models import Document, User
from financial_assistant.models.document import DocumentType
from financial_assistant.services import (
    publish_orphaned_documents,
    purge_deleted_users,
)


@pytest.fixture
def accounts_db(private_database, monkeypatch):
    """Admin scripts on a private schema, plus helpers to create committed rows."""
    factory = private_database.factory
    for module in (publish_orphaned_documents, purge_deleted_users):
        monkeypatch.setattr(module, "async_session_maker", factory)

    async def user(email: str, *, deleted: bool = False) -> int:
        async with factory() as session:
            row = User(
                email=email,
                hashed_password="x",
                is_active=not deleted,
                deleted_at=datetime.now(UTC) if deleted else None,
            )
            session.add(row)
            await session.commit()
            return row.id

    async def document(owner_id: int | None) -> int:
        async with factory() as session:
            row = Document(
                filename="report.pdf",
                company_ticker="TEST",
                document_type=DocumentType.ANNUAL_REPORT,
                year=2026,
                owner_id=owner_id,
            )
            session.add(row)
            await session.commit()
            return row.id

    return SimpleNamespace(factory=factory, user=user, document=document)
