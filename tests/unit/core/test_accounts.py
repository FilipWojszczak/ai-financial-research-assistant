import pytest

from financial_assistant.core.accounts import TOMBSTONE_PASSWORD_HASH, tombstone_email
from financial_assistant.core.security import verify_password

pytestmark = pytest.mark.unit


def test_tombstone_email_is_anonymous_and_unique_per_user():
    assert tombstone_email(7) == "deleted-7@deleted.invalid"
    assert tombstone_email(7) != tombstone_email(8)


@pytest.mark.parametrize("password", ["", "!", "securepassword"])
def test_tombstone_password_hash_never_verifies(password):
    # Must not raise either: pwdlib rejects malformed hashes with an exception.
    assert verify_password(password, TOMBSTONE_PASSWORD_HASH) is False
