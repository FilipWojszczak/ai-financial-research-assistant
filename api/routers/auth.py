from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from ..auth_dependencies import get_current_user
from ..db import get_session
from ..models import User
from ..schemas.auth import Token, UserCreate, UserRead
from ..utils import authenticate_user, create_access_token, hash_password

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=UserRead,
    status_code=201,
    summary="Register a new user account",
    description=(
        "Creates a new user account using email and password.  \n"
        "Validates that the email is unique and stores the password using a secure "
        "hash.\n\n"
        "This endpoint is publicly accessible and does not require authentication."
    ),
)
async def register_user(
    user_data: UserCreate, session: Annotated[AsyncSession, Depends(get_session)]
):
    result = await session.exec(select(User).where(User.email == user_data.email))
    existing_user = result.one_or_none()
    if existing_user:
        raise HTTPException(
            status_code=409, detail="User with this email already exists"
        )

    new_user = User(
        email=user_data.email,
        hashed_password=hash_password(user_data.password),
    )
    session.add(new_user)
    await session.commit()
    await session.refresh(new_user)
    return new_user


@router.post(
    "/token",
    response_model=Token,
    summary="Log in and obtain an access token",
    description=(
        "Authenticates the user using email and password. "
        "Returns a JWT access token that must be included in the `Authorization: "
        "Bearer <token>` header for all protected endpoints.\n\n"
        "This endpoint follows the OAuth2 Password Flow used by FastAPI's built-in "
        "`OAuth2PasswordRequestForm`. Note that `username` in the form corresponds to "
        "the user's email."
    ),
)
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    session: Annotated[AsyncSession, Depends(get_session)],
):
    user = await authenticate_user(form_data.username, form_data.password, session)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(user.id)
    return Token(access_token=access_token)


@router.get(
    "/me",
    response_model=UserRead,
    summary="Retrieve the currently authenticated user",
    description=(
        "Returns basic information about the currently authenticated user: "
        "`id`, `email` and the `is_active` flag."
    ),
)
async def get_my_data(user: Annotated[User, Depends(get_current_user)]):
    return user
