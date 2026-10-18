"""Remove tombstoned users that no longer own any document.

A tombstoned user (`deleted_at IS NOT NULL`) is eligible once it owns no document,
for example after an administrator published all of them. Dry run unless --yes.
"""

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import delete, exists, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.db import async_session_maker, engine
from ..models import Document, User
from .publish_orphaned_documents import positive_int

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List tombstoned users")
    parser.add_argument(
        "--user-id",
        type=positive_int,
        action="append",
        dest="user_ids",
        metavar="ID",
        help="Only purge these users (repeatable); default: every eligible user",
    )
    parser.add_argument(
        "--yes", action="store_true", help="Delete; without it, only a dry run"
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list and (args.yes or args.user_ids):
        parser.error("--list cannot be combined with --yes or --user-id")
    return args


async def _tombstones(session: AsyncSession) -> list[tuple[int, datetime, int]]:
    """Return (id, deleted_at, documents owned) for every tombstoned user."""
    owned = (
        select(func.count())
        .select_from(Document)
        .where(Document.owner_id == User.id)
        .scalar_subquery()
    )
    rows = await session.execute(
        select(User.id, User.deleted_at, owned)
        .where(User.deleted_at.is_not(None))
        .order_by(User.deleted_at, User.id)
    )
    return [(row[0], row[1], row[2]) for row in rows]


async def list_tombstones() -> int:
    async with async_session_maker() as session:
        tombstones = await _tombstones(session)
    if not tombstones:
        print("No tombstoned users.")
        return 0
    print("id\tdeleted at\tdocuments owned")
    for user_id, deleted_at, owned in tombstones:
        marker = "\teligible" if owned == 0 else ""
        print(f"{user_id}\t{deleted_at}\t{owned}{marker}")
    return 0


def refusal_reasons(
    requested_ids: Sequence[int], tombstones: dict[int, int], existing: set[int]
) -> dict[int, str]:
    """Map each requested id that cannot be purged to the reason.

    `tombstones` maps tombstoned user ids to the number of documents they own.
    """
    reasons: dict[int, str] = {}
    for user_id in dict.fromkeys(requested_ids):
        if user_id not in existing:
            reasons[user_id] = "no such user"
        elif user_id not in tombstones:
            reasons[user_id] = "not a deleted account"
        elif tombstones[user_id]:
            reasons[user_id] = f"still owns {tombstones[user_id]} document(s)"
    return reasons


async def _purge(
    session: AsyncSession, requested_ids: list[int] | None, *, apply: bool
) -> int:
    tombstones = {user_id: owned for user_id, _, owned in await _tombstones(session)}
    if requested_ids:
        existing = set(
            await session.scalars(select(User.id).where(User.id.in_(requested_ids)))
        )
        reasons = refusal_reasons(requested_ids, tombstones, existing)
        if reasons:
            for user_id, reason in sorted(reasons.items()):
                print(f"Refused user {user_id}: {reason}")
            print("Nothing was removed.")
            return 1
        ids = sorted(set(requested_ids))
    else:
        ids = sorted(user_id for user_id, owned in tombstones.items() if owned == 0)

    if not ids:
        print("No eligible tombstoned users.")
        return 0
    if not apply:
        print("Dry run; re-run with --yes to remove users: " + ", ".join(map(str, ids)))
        return 0

    try:
        removed = list(
            await session.scalars(
                delete(User)
                .where(
                    User.deleted_at.is_not(None),
                    User.id.in_(ids),
                    ~exists().where(Document.owner_id == User.id),
                )
                .returning(User.id)
            )
        )
        if requested_ids and len(removed) != len(ids):
            # A requested user gained a document after the check.
            await session.rollback()
            kept = sorted(set(ids) - set(removed))
            print(
                "Users gained documents concurrently: "
                + ", ".join(map(str, kept))
                + "; nothing was removed."
            )
            return 1
        await session.commit()
    except IntegrityError:
        # A document appeared meanwhile: the foreign key keeps the user.
        await session.rollback()
        print("A user gained a document concurrently; nothing was removed.")
        return 1
    for user_id in sorted(removed):
        logger.info("Removed tombstoned user %d", user_id)
    print(f"Removed {len(removed)} user(s).")
    return 0


async def purge(requested_ids: list[int] | None, *, apply: bool) -> int:
    async with async_session_maker() as session:
        return await _purge(session, requested_ids, apply=apply)


async def run(args: argparse.Namespace) -> int:
    try:
        if args.list:
            return await list_tombstones()
        return await purge(args.user_ids, apply=args.yes)
    finally:
        await engine.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
