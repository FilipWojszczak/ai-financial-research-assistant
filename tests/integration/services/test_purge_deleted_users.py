import pytest
from sqlalchemy import select

from financial_assistant.models import User
from financial_assistant.services.publish_orphaned_documents import publish
from financial_assistant.services.purge_deleted_users import list_tombstones, purge

pytestmark = pytest.mark.integration


async def existing_users(db):
    async with db.factory() as session:
        return set(await session.scalars(select(User.id)))


@pytest.fixture
async def users(accounts_db):
    empty = await accounts_db.user("deleted-a@deleted.invalid", deleted=True)
    owning = await accounts_db.user("deleted-b@deleted.invalid", deleted=True)
    live = await accounts_db.user("live@example.com")
    document = await accounts_db.document(owning)
    return {"empty": empty, "owning": owning, "live": live, "document": document}


async def test_list_marks_eligible_tombstones(users, capsys):
    assert await list_tombstones() == 0
    lines = capsys.readouterr().out.splitlines()[1:]
    by_id = {int(line.split("\t")[0]): line for line in lines}
    assert set(by_id) == {users["empty"], users["owning"]}
    assert by_id[users["empty"]].endswith("\t0\teligible")
    assert by_id[users["owning"]].endswith("\t1")


async def test_dry_run_changes_nothing(accounts_db, users, capsys):
    before = await existing_users(accounts_db)
    assert await purge(None, apply=False) == 0
    assert str(users["empty"]) in capsys.readouterr().out
    assert await existing_users(accounts_db) == before


async def test_removes_only_tombstones_without_documents(accounts_db, users):
    assert await purge(None, apply=True) == 0
    remaining = await existing_users(accounts_db)
    assert users["empty"] not in remaining
    assert {users["owning"], users["live"]} <= remaining


@pytest.mark.parametrize("invalid", ["owning", "live", "unknown"])
async def test_ineligible_user_id_refuses_the_whole_batch(
    accounts_db, users, invalid, capsys
):
    before = await existing_users(accounts_db)
    other = users.get(invalid, 999_999)

    assert await purge([users["empty"], other], apply=True) == 1

    assert "Nothing was removed." in capsys.readouterr().out
    assert await existing_users(accounts_db) == before


async def test_purge_after_publishing_the_last_document(accounts_db, users):
    assert await publish([users["document"]], apply=True) == 0
    assert await purge([users["owning"]], apply=True) == 0
    remaining = await existing_users(accounts_db)
    assert users["owning"] not in remaining
    assert {users["empty"], users["live"]} <= remaining
