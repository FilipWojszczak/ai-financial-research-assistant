from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from langchain_core.messages.ai import UsageMetadata
from sqlalchemy import delete, select, update

from financial_assistant.agents.chat.turns import (
    ThreadBusyError,
    claim_turn,
    finish_turn,
    load_scope_documents,
    thread_title,
    tokens_used_today,
)
from financial_assistant.models import ChatThread, ChatTurn, Document, User
from financial_assistant.models.chat import ChatTurnStatus
from financial_assistant.models.document import DocumentStatus, DocumentType
from financial_assistant.retrieval.types import RetrievalScope

pytestmark = pytest.mark.integration

_TURNS_MODULE = "financial_assistant.agents.chat.turns"
STALE_AFTER = timedelta(seconds=360)
USAGE: dict[str, UsageMetadata] = {
    "model-a": {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
    "model-b": {"input_tokens": 50, "output_tokens": 5, "total_tokens": 55},
}


async def _user(factory, email="reader@example.com") -> int:
    async with factory() as session:
        user = User(email=email, hashed_password="unused")
        session.add(user)
        await session.commit()
        return user.id


async def _thread(factory, user_id, **fields) -> ChatThread:
    async with factory() as session:
        thread = ChatThread(user_id=user_id, **fields)
        session.add(thread)
        await session.commit()
        return thread


async def _claim(factory, thread: ChatThread) -> ChatTurn:
    async with factory() as session:
        return await claim_turn(
            session,
            thread_id=thread.id,
            user_id=thread.user_id,
            stale_after=STALE_AFTER,
        )


async def _get(factory, model, id_):
    async with factory() as session:
        return await session.get(model, id_)


async def test_a_second_claim_on_the_same_thread_is_refused(private_database):
    factory = private_database.factory
    user_id = await _user(factory)
    thread = await _thread(factory, user_id)
    first = await _claim(factory, thread)
    assert first.status == ChatTurnStatus.RUNNING
    with pytest.raises(ThreadBusyError):
        await _claim(factory, thread)
    assert (await _get(factory, ChatTurn, first.id)).status == ChatTurnStatus.RUNNING


async def test_claims_on_different_threads_and_after_finishing_succeed(
    private_database,
):
    factory = private_database.factory
    user_id = await _user(factory)
    thread = await _thread(factory, user_id)
    other = await _thread(factory, user_id)
    turn = await _claim(factory, thread)
    await _claim(factory, other)
    await finish_turn(
        factory, turn.id, status=ChatTurnStatus.COMPLETED, usage={}, error_code=None
    )
    await _claim(factory, thread)


async def test_a_stale_running_turn_is_abandoned_and_the_new_claim_succeeds(
    private_database,
):
    factory = private_database.factory
    thread = await _thread(factory, await _user(factory))
    stale = await _claim(factory, thread)
    async with factory() as session:
        await session.execute(
            update(ChatTurn)
            .where(ChatTurn.id == stale.id)
            .values(started_at=datetime.now(UTC) - STALE_AFTER - timedelta(seconds=1))
        )
        await session.commit()

    fresh = await _claim(factory, thread)
    abandoned = await _get(factory, ChatTurn, stale.id)
    assert abandoned.status == ChatTurnStatus.FAILED
    assert abandoned.error_code == "abandoned"
    assert abandoned.finished_at is not None
    assert (await _get(factory, ChatTurn, fresh.id)).status == ChatTurnStatus.RUNNING


async def test_finish_records_usage_and_touches_the_thread(private_database):
    factory = private_database.factory
    thread = await _thread(factory, await _user(factory))
    turn = await _claim(factory, thread)
    await finish_turn(
        factory,
        turn.id,
        status=ChatTurnStatus.COMPLETED,
        usage=USAGE,
        error_code=None,
        title="First question",
    )
    finished = await _get(factory, ChatTurn, turn.id)
    assert finished.status == ChatTurnStatus.COMPLETED
    assert (finished.input_tokens, finished.output_tokens) == (150, 15)
    assert finished.usage == USAGE
    assert finished.finished_at is not None
    touched = await _get(factory, ChatThread, thread.id)
    assert touched.title == "First question"
    assert touched.updated_at > thread.updated_at


async def test_finish_keeps_an_existing_title(private_database):
    factory = private_database.factory
    thread = await _thread(factory, await _user(factory), title="Chosen title")
    turn = await _claim(factory, thread)
    await finish_turn(
        factory,
        turn.id,
        status=ChatTurnStatus.FAILED,
        usage={},
        error_code="timeout",
        title="Later question",
    )
    assert (await _get(factory, ChatThread, thread.id)).title == "Chosen title"
    assert (await _get(factory, ChatTurn, turn.id)).error_code == "timeout"


def test_thread_title_collapses_whitespace_and_truncates():
    assert thread_title("  What   was\n revenue? ") == "What was revenue?"
    assert len(thread_title("x" * 200)) == 80


async def test_tokens_used_today_counts_only_today_and_only_this_user(private_database):
    factory = private_database.factory
    user_id = await _user(factory)
    other_id = await _user(factory, "other@example.com")
    now = datetime.now(UTC)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    async with factory() as session:
        assert await tokens_used_today(session, user_id) == 0
        session.add_all(
            [
                ChatTurn(
                    user_id=user_id,
                    status=ChatTurnStatus.COMPLETED,
                    input_tokens=100,
                    output_tokens=20,
                    started_at=now,
                ),
                ChatTurn(
                    user_id=user_id,
                    status=ChatTurnStatus.FAILED,
                    input_tokens=30,
                    output_tokens=0,
                    started_at=midnight,
                ),
                ChatTurn(
                    user_id=user_id,
                    status=ChatTurnStatus.COMPLETED,
                    input_tokens=1000,
                    output_tokens=1000,
                    started_at=midnight - timedelta(seconds=1),
                ),
                ChatTurn(
                    user_id=other_id,
                    status=ChatTurnStatus.COMPLETED,
                    input_tokens=500,
                    output_tokens=500,
                    started_at=now,
                ),
            ]
        )
        await session.commit()
        assert await tokens_used_today(session, user_id) == 150


async def test_deleting_a_thread_keeps_its_turns_for_the_budget(private_database):
    factory = private_database.factory
    user_id = await _user(factory)
    thread = await _thread(factory, user_id)
    turn = await _claim(factory, thread)
    await finish_turn(
        factory, turn.id, status=ChatTurnStatus.COMPLETED, usage=USAGE, error_code=None
    )
    async with factory() as session:
        await session.execute(delete(ChatThread).where(ChatThread.id == thread.id))
        await session.commit()
    async with factory() as session:
        kept = await session.scalar(select(ChatTurn).where(ChatTurn.id == turn.id))
        assert kept.thread_id is None
        assert await tokens_used_today(session, user_id) == 165


async def test_scope_documents_are_the_accessible_completed_ones(private_database):
    factory = private_database.factory
    user_id = await _user(factory)
    other_id = await _user(factory, "other@example.com")

    def document(owner_id, status=DocumentStatus.COMPLETED):
        return Document(
            filename="a.pdf",
            company_ticker="ACME",
            document_type=DocumentType.ANNUAL_REPORT,
            year=2025,
            owner_id=owner_id,
            status=status,
        )

    public, own = document(None), document(user_id)
    async with factory() as session:
        session.add_all(
            [
                public,
                own,
                document(other_id),
                document(user_id, DocumentStatus.PROCESSING),
            ]
        )
        await session.commit()
        scope = RetrievalScope(user_id=user_id)
        documents, unlisted = await load_scope_documents(session, scope)
        assert {item.id for item in documents} == {public.id, own.id}
        assert unlisted == 0
        assert documents[0].company_ticker == "ACME"
        assert documents[0].document_type == DocumentType.ANNUAL_REPORT
        empty = RetrievalScope(user_id=user_id, company_ticker="NONE")
        assert await load_scope_documents(session, empty) == ((), 0)

        # Beyond the cap, documents are counted rather than listed.
        with patch(f"{_TURNS_MODULE}._MAX_SCOPE_DOCUMENTS", 1):
            documents, unlisted = await load_scope_documents(session, scope)
        assert [item.id for item in documents] == [min(public.id, own.id)]
        assert unlisted == 1
