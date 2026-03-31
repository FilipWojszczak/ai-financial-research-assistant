from typing import Protocol

from api.models import User


class UserFactory(Protocol):
    async def __call__(self, email: str, password: str = "securepassword") -> User: ...  # noqa: S107


class TokenFactory(Protocol):
    def __call__(self, user: User) -> str: ...
