from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from ...ai.document_ingestion import process_uploaded_document
from ...core.db import get_session
from ...models import Document, User
from ...schemas.document import DocumentCreate, DocumentRead
from ..dependencies.auth import get_current_user

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
    document_data: Annotated[DocumentCreate, Form()],
    file: Annotated[UploadFile, File()],
    background_tasks: BackgroundTasks,
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
):
    owner_id = None if document_data.is_public else user.id

    db_document = Document(
        filename=document_data.filename,
        company_ticker=document_data.company_ticker,
        document_type=document_data.document_type,
        year=document_data.year,
        owner_id=owner_id,
    )
    session.add(db_document)
    await session.commit()
    await session.refresh(db_document)

    file_bytes = await file.read()

    background_tasks.add_task(
        process_uploaded_document, document_id=db_document.id, file_bytes=file_bytes
    )

    return db_document
