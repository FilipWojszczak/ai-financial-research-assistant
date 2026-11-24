import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from financial_assistant.core import accounts
from financial_assistant.core.accounts import (
    AccountBusyError,
    AccountChangedError,
    ChatBusyError,
    delete_account,
)
from financial_assistant.models import ChatThread, ChatTurn, Document, User
from financial_assistant.models.chat import ChatTurnStatus
from financial_assistant.models.document import DocumentType

pytestmark = pytest.mark.integration


async def _chatting_user(
    factory,
    *,
    owns_document: bool = False,
    status: ChatTurnStatus = ChatTurnStatus.COMPLETED,
    started_at: datetime | None = None,
) -> tuple[int, ChatThread]:
    async with factory() as session:
        user = User(email="chatter@example.com", hashed_password="x")
        session.add(user)
        await session.flush()
        thread = ChatThread(user_id=user.id)
        session.add(thread)
        await session.flush()
        session.add(
            ChatTurn(
                thread_id=thread.id,
                user_id=user.id,
                status=status,
                started_at=started_at or datetime.now(UTC),
            )
        )
        if owns_document:
            session.add(
                Document(
                    filename="a.pdf",
                    company_ticker="AAPL",
                    document_type=DocumentType.ANNUAL_REPORT,
                    year=2026,
                    owner_id=user.id,
                )
            )
        await session.commit()
        return user.id, thread


async def _delete(factory, user_id: int, checkpointer) -> None:
    async with factory() as session:
        user = await session.get(User, user_id)
        await delete_account(session, user, checkpointer=checkpointer)
        await session.commit()


async def _chat_rows(factory) -> tuple[int, int]:
    async with factory() as session:
        threads = await session.scalar(select(func.count()).select_from(ChatThread))
        turns = await session.scalar(select(func.count()).select_from(ChatTurn))
        return threads, turns


@pytest.mark.parametrize("owns_document", [False, True], ids=["hard", "tombstone"])
async def test_chat_data_is_deleted_with_the_account(private_database, owns_document):
    factory = private_database.factory
    user_id, thread = await _chatting_user(factory, owns_document=owns_document)
    checkpointer = AsyncMock()

    await _delete(factory, user_id, checkpointer)

    checkpointer.adelete_thread.assert_awaited_once_with(str(thread.id))
    assert await _chat_rows(factory) == (0, 0)
    async with factory() as check:
        user = await check.get(User, user_id)
        assert (user is not None) == owns_document


async def test_a_running_turn_blocks_the_deletion(private_database):
    factory = private_database.factory
    user_id, _ = await _chatting_user(factory, status=ChatTurnStatus.RUNNING)
    checkpointer = AsyncMock()

    with pytest.raises(ChatBusyError):
        await _delete(factory, user_id, checkpointer)

    checkpointer.adelete_thread.assert_not_awaited()
    assert await _chat_rows(factory) == (1, 1)


async def test_an_abandoned_turn_does_not_block_the_deletion(private_database):
    factory = private_database.factory
    user_id, _ = await _chatting_user(
        factory,
        status=ChatTurnStatus.RUNNING,
        started_at=datetime.now(UTC) - timedelta(hours=1),
    )
    await _delete(factory, user_id, AsyncMock())
    assert await _chat_rows(factory) == (0, 0)


async def test_a_turn_started_while_waiting_for_a_document_blocks_the_deletion(
    private_database,
):
    factory = private_database.factory
    user_id, _ = await _chatting_user(factory, owns_document=True)

    async def delete_with_documents():
        async with factory() as session:
            user = await session.get(User, user_id)
            await delete_account(session, user, delete_documents=True)

    async with factory() as worker, factory() as chat:
        await worker.scalars(select(Document.id).with_for_update())
        task = asyncio.create_task(delete_with_documents())
        await asyncio.sleep(0.5)
        assert not task.done()  # Waits for the worker's lock.
        thread = ChatThread(user_id=user_id)
        chat.add(thread)
        await chat.flush()
        chat.add(ChatTurn(thread_id=thread.id, user_id=user_id))
        await asyncio.wait_for(chat.commit(), timeout=2)
        await worker.rollback()
        with pytest.raises(ChatBusyError):
            await asyncio.wait_for(task, timeout=5)
    assert await _chat_rows(factory) == (2, 2)


async def test_a_lock_held_by_another_request_is_not_blamed_on_documents(
    private_database, monkeypatch
):
    monkeypatch.setattr(accounts, "LOCK_TIMEOUT", "100ms")
    factory = private_database.factory
    user_id, thread = await _chatting_user(factory)
    async with factory() as other:
        await other.get(ChatThread, thread.id, with_for_update=True)
        with pytest.raises(AccountBusyError):
            await _delete(factory, user_id, AsyncMock())
    assert await _chat_rows(factory) == (1, 1)


async def test_a_checkpoint_failure_keeps_the_account_for_a_retry(private_database):
    factory = private_database.factory
    user_id, _ = await _chatting_user(factory)
    failing = AsyncMock()
    failing.adelete_thread.side_effect = ConnectionError("pool is down")

    with pytest.raises(ConnectionError):
        await _delete(factory, user_id, failing)
    async with factory() as check:
        assert await check.get(User, user_id) is not None
    assert await _chat_rows(factory) == (1, 1)

    await _delete(factory, user_id, AsyncMock())
    assert await _chat_rows(factory) == (0, 0)


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
