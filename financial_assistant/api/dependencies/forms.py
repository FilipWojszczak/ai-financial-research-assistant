from datetime import datetime
from typing import Annotated

from fastapi import Form

from ...models.document import DocumentType
from ...schemas.document import DocumentCreate


async def document_create_form(
    company_ticker: Annotated[str, Form()],
    document_type: Annotated[DocumentType, Form()],
    year: Annotated[int | None, Form()] = None,
    is_public: Annotated[bool, Form()] = False,
) -> DocumentCreate:
    return DocumentCreate(
        company_ticker=company_ticker,
        document_type=document_type,
        year=year if year is not None else datetime.now().year,
        is_public=is_public,
    )
