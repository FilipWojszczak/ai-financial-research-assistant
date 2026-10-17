from datetime import UTC, datetime

import pytest

from financial_assistant.services.publish_orphaned_documents import (
    parse_args,
    printable,
    refusal_reasons,
)

pytestmark = pytest.mark.unit

DELETED_AT = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--yes"],
        ["--list", "--document-id", "1"],
        ["--list", "--yes"],
        ["--document-id", "0"],
        ["--document-id", "-3"],
        ["--document-id", "abc"],
    ],
)
def test_invalid_arguments_are_usage_errors(argv):
    with pytest.raises(SystemExit) as exc:
        parse_args(argv)
    assert exc.value.code == 2


def test_valid_arguments():
    assert parse_args(["--list"]).list is True
    args = parse_args(["--document-id", "12", "--document-id", "15", "--yes"])
    assert (args.document_ids, args.yes, args.list) == ([12, 15], True, False)
    assert parse_args(["--document-id", "12"]).yes is False


def test_refusal_reasons_names_every_offending_id():
    found = {
        1: (5, DELETED_AT),  # orphaned
        2: (None, None),  # public
        3: (6, None),  # live owner
        4: (5, DELETED_AT),  # orphaned, but requested twice
    }
    assert refusal_reasons([1, 2, 3, 4, 4, 9], found) == {
        2: "already public",
        3: "owned by an active account",
        4: "requested more than once",
        9: "no such document",
    }


def test_refusal_reasons_accepts_orphaned_documents():
    assert refusal_reasons([1], {1: (5, DELETED_AT)}) == {}


def test_printable_neutralizes_control_characters():
    assert printable("a.pdf\n1\tAAPL\x1b[2J") == "a.pdf?1?AAPL?[2J"
    assert printable("Raport roczny ż.pdf") == "Raport roczny ż.pdf"
