from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
)
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ...ai.document_ingestion import process_uploaded_document
from ...core.db import get_session
from ...models import Document, User
from ...schemas.document import DocumentCreate, DocumentRead
from ..dependencies.auth import get_current_user
from ..dependencies.forms import document_create_form

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/",
    response_model=DocumentRead,
    # 202 = Accepted - indicates that the request has been accepted for processing, but
    # the processing is not complete
    status_code=202,
    summary="Upload a new document for processing",
    description=(
        "Upload a financial document (e.g., 10-K, 10-Q) for a specific company. \n"
        "The document will be processed in the background. The response will include "
        "the document metadata, and the status will indicate that the document is "
        "being processed. Once processing is complete, the status will be updated "
        "accordingly.\n\n"
        "If is_public is set, the document will be accessible to all users; otherwise, "
        "it will be private to the uploading user."
    ),
)
async def upload_document(
    document_data: Annotated[DocumentCreate, Depends(document_create_form)],
    file: Annotated[UploadFile, File(type="application/pdf")],
    background_tasks: BackgroundTasks,
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
):
    owner_id = None if document_data.is_public else user.id

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=422, detail="Only PDF files are allowed")
    if not file.filename:
        raise HTTPException(status_code=422, detail="File must have a filename")

    file_bytes = await file.read()

    db_document = Document(
        filename=file.filename,
        company_ticker=document_data.company_ticker,
        document_type=document_data.document_type,
        year=document_data.year,
        owner_id=owner_id,
    )
    session.add(db_document)
    await session.commit()
    await session.refresh(db_document)

    background_tasks.add_task(
        process_uploaded_document, document_id=db_document.id, file_bytes=file_bytes
    )

    return db_document


@router.get(
    "/",
    response_model=list[DocumentRead],
    summary="List accessible documents",
    description="Returns all public documents and documents owned by the current user.",
)
async def list_documents(
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
):
    result = await session.execute(
        select(Document).where(
            or_(Document.owner_id.is_(None), Document.owner_id == user.id)
        )
    )
    return result.scalars().all()


@router.get(
    "/{document_id}",
    response_model=DocumentRead,
    summary="Get a document by ID",
    description=(
        "Returns a document if it is public or owned by the current user. "
        "Returns 404 if the document does not exist or is not accessible."
    ),
)
async def get_document(
    document_id: int,
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
):
    result = await session.execute(
        select(Document).where(
            Document.id == document_id,
            or_(Document.owner_id.is_(None), Document.owner_id == user.id),
        )
    )
    document = result.scalar_one_or_none()
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.delete(
    "/{document_id}",
    status_code=204,
    summary="Delete a document",
    description=(
        "Deletes a document owned by the current user. "
        "Returns 404 if the document does not exist or is not accessible. "
        "Returns 403 if the document is public and cannot be deleted."
    ),
)
async def delete_document(
    document_id: int,
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
):
    result = await session.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    if document is not None and document.owner_id is None:
        raise HTTPException(
            status_code=403, detail="Public documents cannot be deleted"
        )
    if document is None or document.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Document not found")
    await session.delete(document)
    await session.commit()
