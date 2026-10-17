import asyncio

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from financial_assistant.core.accounts import AccountChangedError, delete_account
from financial_assistant.models import Document, User
from financial_assistant.models.document import DocumentType

pytestmark = pytest.mark.integration


async def test_deleting_a_user_who_owns_documents_fails(private_database):
    # passive_deletes="all": the foreign key refuses instead of nulling owner_id,
    # which would have made the documents public.
    factory = private_database.factory
    async with factory() as setup:
        user = User(email="owner@example.com", hashed_password="x")
        setup.add(user)
        await setup.flush()
        document = Document(
            filename="a.pdf",
            company_ticker="AAPL",
            document_type=DocumentType.ANNUAL_REPORT,
            year=2026,
            owner_id=user.id,
        )
        setup.add(document)
        await setup.commit()
        user_id, document_id = user.id, document.id

    async with factory() as session:
        user = await session.get(User, user_id)
        await session.delete(user)
        with pytest.raises(IntegrityError):
            await session.commit()

    async with factory() as check:
        assert (
            await check.scalar(
                select(Document.owner_id).where(Document.id == document_id)
            )
            == user_id
        )


async def test_upload_racing_account_deletion_keeps_the_user(private_database):
    # The upload is not visible to the ownership check yet, so the user DELETE waits
    # on its FK lock and fails once it commits: a retryable error, never a 500.
    factory = private_database.factory
    async with factory() as setup:
        user = User(email="racer@example.com", hashed_password="x")
        setup.add(user)
        await setup.commit()
        user_id = user.id

    async with factory() as upload, factory() as deletion:
        upload.add(
            Document(
                filename="late.pdf",
                company_ticker="AAPL",
                document_type=DocumentType.ANNUAL_REPORT,
                year=2026,
                owner_id=user_id,
            )
        )
        await upload.flush()
        target = await deletion.get(User, user_id)
        assert target is not None
        task = asyncio.create_task(delete_account(deletion, target))
        await asyncio.sleep(0.5)
        assert not task.done()  # Blocked on the upload's lock on the user row.
        await upload.commit()
        with pytest.raises(AccountChangedError):
            await asyncio.wait_for(task, timeout=5)
        await deletion.rollback()

    async with factory() as check:
        assert await check.scalar(select(User.id).where(User.id == user_id)) == user_id
