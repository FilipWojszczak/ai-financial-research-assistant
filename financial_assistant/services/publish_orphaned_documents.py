"""List documents of deleted accounts and make chosen ones public.

A document is orphaned when its owner was tombstoned (`user.deleted_at IS NOT NULL`).
Publishing is a dry run unless --yes is given, and all-or-nothing.
"""

import argparse
import asyncio
import logging
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import Row, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.db import async_session_maker, engine
from ..models import Document, User

logger = logging.getLogger(__name__)


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", help="List orphaned documents")
    mode.add_argument(
        "--document-id",
        type=positive_int,
        action="append",
        dest="document_ids",
        metavar="ID",
        help="Document to publish (repeatable)",
    )
    parser.add_argument(
        "--yes", action="store_true", help="Publish; without it, only a dry run"
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list and args.yes:
        parser.error("--yes cannot be used with --list")
    return args


def refusal_reasons(
    requested_ids: Sequence[int],
    found: dict[int, tuple[int | None, datetime | None]],
) -> dict[int, str]:
    """Map each id that must not be published to the reason.

    `found` maps existing document ids to (owner id, owner's deleted_at).
    """
    reasons: dict[int, str] = {}
    for document_id, count in Counter(requested_ids).items():
        if count > 1:
            reasons[document_id] = "requested more than once"
        elif document_id not in found:
            reasons[document_id] = "no such document"
        elif found[document_id][0] is None:
            reasons[document_id] = "already public"
        elif found[document_id][1] is None:
            reasons[document_id] = "owned by an active account"
    return reasons


def _orphaned_query():
    return (
        select(
            Document.id,
            Document.company_ticker,
            Document.document_type,
            Document.year,
            Document.filename,
            Document.status,
            User.id.label("owner_id"),
            User.deleted_at,
        )
        .join(User, Document.owner_id == User.id)
        .where(User.deleted_at.is_not(None))
        .order_by(User.deleted_at, Document.id)
    )


def printable(value: str) -> str:
    """Replace control characters (newlines, tabs, ANSI escapes) in user input."""
    return "".join(char if char.isprintable() else "?" for char in value)


def _describe(row: Row) -> str:
    # Filenames and tickers come from users: keep them from forging output rows.
    return (
        f"{row.id}\t{printable(row.company_ticker)}\t{row.document_type}\t"
        f"{row.year}\t{printable(row.filename)}\t{row.status}\towner={row.owner_id}\t"
        f"deleted_at={row.deleted_at.isoformat()}"
    )


async def list_orphaned() -> int:
    async with async_session_maker() as session:
        rows = (await session.execute(_orphaned_query())).all()
    if not rows:
        print("No orphaned documents.")
        return 0
    print("id\tticker\ttype\tyear\tfilename\tstatus\tformer owner\tdeleted at")
    for row in rows:
        print(_describe(row))
    return 0


async def _publish(session: AsyncSession, ids: list[int], *, apply: bool) -> int:
    found = {
        row.id: (row.owner_id, row.deleted_at)
        for row in await session.execute(
            select(Document.id, Document.owner_id, User.deleted_at)
            .outerjoin(User, Document.owner_id == User.id)
            .where(Document.id.in_(ids))
        )
    }
    reasons = refusal_reasons(ids, found)
    if reasons:
        for document_id, reason in sorted(reasons.items()):
            print(f"Refused document {document_id}: {reason}")
        print("Nothing was published.")
        return 1

    rows = (await session.execute(_orphaned_query().where(Document.id.in_(ids)))).all()
    if not apply:
        print("Dry run; re-run with --yes to publish:")
        for row in rows:
            print(_describe(row))
        return 0

    tombstoned = select(User.id).where(User.deleted_at.is_not(None))
    result = await session.execute(
        update(Document)
        .where(Document.id.in_(ids), Document.owner_id.in_(tombstoned))
        .values(owner_id=None)
    )
    if result.rowcount != len(ids):  # pyright: ignore[reportAttributeAccessIssue]
        await session.rollback()
        print("The documents changed concurrently; nothing was published.")
        return 1
    await session.commit()
    for row in rows:
        logger.info(
            "Published orphaned document %d (former owner %d)", row.id, row.owner_id
        )
    print(f"Published {len(ids)} document(s).")
    return 0


async def publish(ids: list[int], *, apply: bool) -> int:
    async with async_session_maker() as session:
        return await _publish(session, ids, apply=apply)


async def run(args: argparse.Namespace) -> int:
    try:
        if args.list:
            return await list_orphaned()
        return await publish(args.document_ids, apply=args.yes)
    finally:
        await engine.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
