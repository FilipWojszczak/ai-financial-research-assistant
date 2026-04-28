from datetime import UTC, datetime, timedelta

import jwt
from pwdlib import PasswordHash
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .core.config import get_settings
from .models import User

settings = get_settings()
SECRET_KEY = settings.secret_key
ALGORITHM = settings.algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = 60


password_hash = PasswordHash.recommended()


class InvalidTokenError(Exception):
    """Raised when JWT is invalid or cannot be used"""


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return password_hash.verify(plain_password, hashed_password)


async def authenticate_user(
    email: str, password: str, session: AsyncSession
) -> User | None:
    result = await session.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if (
        not user
        or not verify_password(password, user.hashed_password)
        or not user.is_active
    ):
        return None
    return user


def create_access_token(
    user_id: int, expires_delta: timedelta | int = ACCESS_TOKEN_EXPIRE_MINUTES
) -> str:
    to_encode = {"sub": str(user_id), "iat": datetime.now(UTC)}
    if isinstance(expires_delta, timedelta):
        expire = to_encode["iat"] + expires_delta
    else:
        expire = to_encode["iat"] + timedelta(minutes=expires_delta)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_access_token(token: str) -> int:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        # Token valid, but expired
        raise InvalidTokenError("Token has expired") from exc
    except jwt.InvalidTokenError as exc:
        # Token broken or otherwise invalid
        raise InvalidTokenError("Invalid token") from exc

    user_id = payload.get("sub")
    if user_id is None:
        # Token does not contain user identification
        raise InvalidTokenError("Token missing subject (sub)")

    try:
        return int(user_id)
    except (TypeError, ValueError) as exc:
        raise InvalidTokenError("Invalid subject (sub)") from exc
