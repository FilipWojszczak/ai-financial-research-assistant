import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.accounts import AccountChangedError, DocumentsBusyError, delete_account
from ...core.db import get_session
from ...core.document_storage import delete_document_file
from ...core.security import authenticate_user, create_access_token, hash_password
from ...models import User
from ...schemas.auth import AccountDelete, PasswordUpdate, Token, UserCreate, UserRead
from ..dependencies.auth import get_current_user

logger = logging.getLogger(__name__)
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
    result = await session.execute(select(User).where(User.email == user_data.email))
    existing_user = result.scalar_one_or_none()
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


@router.put(
    "/me/password",
    response_model=UserRead,
    summary="Update the currently authenticated user's password",
    description=(
        "Updates the password for the currently authenticated user. "
        "Requires the current password and the new password."
    ),
)
async def update_password(
    payload: PasswordUpdate,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
):
    if not await authenticate_user(user.email, payload.current_password, session):
        raise HTTPException(
            status_code=401,
            detail="Current password is incorrect",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if payload.current_password == payload.new_password:
        raise HTTPException(
            status_code=400,
            detail="Current password and new password cannot be the same",
        )
    user.hashed_password = hash_password(payload.new_password)
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


@router.delete(
    "/me",
    status_code=204,
    summary="Delete the currently authenticated user's account",
    description=(
        "Deletes the account of the currently authenticated user. "
        "This action is irreversible.\n\n"
        "By default, the documents the user uploaded are retained and become "
        "inaccessible to everyone until an administrator publishes them. With "
        "`delete_documents=true` they are deleted together with the account; this "
        "returns 409 and changes nothing while one of them is still being processed."
    ),
)
async def delete_my_account(
    payload: AccountDelete,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
):
    if not await authenticate_user(user.email, payload.password, session):
        raise HTTPException(
            status_code=401,
            detail="Incorrect password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        deleted_ids = await delete_account(
            session, user, delete_documents=payload.delete_documents
        )
        await session.commit()
    except DocumentsBusyError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=409,
            detail=(
                "A document is still being processed; try again later, or delete "
                "the account without deleting documents"
            ),
        ) from exc
    except AccountChangedError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=409,
            detail="A document was uploaded during account deletion; try again",
        ) from exc

    # The account is already gone: a missed file is removed later by cleanup_storage.
    for document_id in deleted_ids:
        try:
            await delete_document_file(document_id)
        except Exception as exc:
            logger.warning(
                "Could not remove the PDF of deleted document %d (%s)",
                document_id,
                type(exc).__name__,
            )
