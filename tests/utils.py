from typing import Protocol

from financial_assistant.models import User


class UserFactory(Protocol):
    async def __call__(self, email: str, password: str = "securepassword") -> User: ...


class TokenFactory(Protocol):
    def __call__(self, user: User) -> str: ...
