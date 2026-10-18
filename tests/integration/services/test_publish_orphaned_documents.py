import pytest
from sqlalchemy import or_, select

from financial_assistant.models import Document
from financial_assistant.services.publish_orphaned_documents import (
    list_orphaned,
    publish,
)

pytestmark = pytest.mark.integration


async def owners(db, ids):
    async with db.factory() as session:
        rows = await session.execute(
            select(Document.id, Document.owner_id).where(Document.id.in_(ids))
        )
        return dict(rows.tuples().all())


async def visible_to(db, user_id):
    async with db.factory() as session:
        return set(
            await session.scalars(
                select(Document.id).where(
                    or_(Document.owner_id.is_(None), Document.owner_id == user_id)
                )
            )
        )


@pytest.fixture
async def orphans(accounts_db):
    tombstone = await accounts_db.user("deleted-1@deleted.invalid", deleted=True)
    live = await accounts_db.user("live@example.com")
    return {
        "tombstone": tombstone,
        "live": live,
        "orphaned": [await accounts_db.document(tombstone) for _ in range(3)],
        "owned": await accounts_db.document(live),
        "public": await accounts_db.document(None),
    }


async def test_list_shows_only_orphaned_documents(orphans, capsys):
    assert await list_orphaned() == 0
    listed = [
        int(line.split("\t")[0]) for line in capsys.readouterr().out.splitlines()[1:]
    ]
    assert listed == orphans["orphaned"]


async def test_list_without_orphans(accounts_db, capsys):
    assert await list_orphaned() == 0
    assert capsys.readouterr().out.strip() == "No orphaned documents."


async def test_dry_run_changes_nothing(accounts_db, orphans, capsys):
    chosen = orphans["orphaned"][:2]
    assert await publish(chosen, apply=False) == 0
    assert "Dry run" in capsys.readouterr().out
    assert set((await owners(accounts_db, chosen)).values()) == {orphans["tombstone"]}


async def test_publishes_only_the_chosen_documents(accounts_db, orphans, caplog):
    chosen, kept = orphans["orphaned"][:2], orphans["orphaned"][2]
    assert orphans["orphaned"][0] not in await visible_to(accounts_db, orphans["live"])

    with caplog.at_level("INFO"):
        assert await publish(chosen, apply=True) == 0

    assert await owners(accounts_db, [*chosen, kept]) == {
        chosen[0]: None,
        chosen[1]: None,
        kept: orphans["tombstone"],
    }
    visible = await visible_to(accounts_db, orphans["live"])
    assert set(chosen) <= visible
    assert kept not in visible
    for document_id in chosen:
        assert f"Published orphaned document {document_id}" in caplog.text


@pytest.mark.parametrize("invalid", ["unknown", "public", "owned", "duplicate"])
async def test_any_invalid_id_refuses_the_whole_batch(
    accounts_db, orphans, invalid, capsys
):
    valid = orphans["orphaned"][0]
    extra = {
        "unknown": [999_999],
        "public": [orphans["public"]],
        "owned": [orphans["owned"]],
        "duplicate": [valid],
    }[invalid]
    before = await owners(accounts_db, [valid, *extra])

    assert await publish([valid, *extra], apply=True) == 1

    assert "Nothing was published." in capsys.readouterr().out
    assert await owners(accounts_db, [valid, *extra]) == before
