import pytest

from financial_assistant.services.purge_deleted_users import parse_args, refusal_reasons

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "argv",
    [
        ["--list", "--yes"],
        ["--list", "--user-id", "1"],
        ["--user-id", "0"],
        ["--user-id", "x"],
    ],
)
def test_invalid_arguments_are_usage_errors(argv):
    with pytest.raises(SystemExit) as exc:
        parse_args(argv)
    assert exc.value.code == 2


def test_valid_arguments():
    assert parse_args([]).yes is False
    assert parse_args(["--list"]).list is True
    args = parse_args(["--user-id", "7", "--user-id", "8", "--yes"])
    assert (args.user_ids, args.yes) == ([7, 8], True)


def test_refusal_reasons():
    tombstones = {1: 0, 2: 3}
    assert refusal_reasons([1, 2, 3, 4], tombstones, existing={1, 2, 3}) == {
        2: "still owns 3 document(s)",
        3: "not a deleted account",
        4: "no such user",
    }
