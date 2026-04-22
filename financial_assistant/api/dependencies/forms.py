from datetime import datetime
from typing import Annotated

from fastapi import Form

from ...models.document import DocumentType
from ...schemas.document import DocumentCreate


async def document_create_form(
    company_ticker: Annotated[str, Form()],
    document_type: Annotated[DocumentType, Form()],
    year: Annotated[int, Form()] = datetime.now().year,
    is_public: Annotated[bool, Form()] = False,
) -> DocumentCreate:
    return DocumentCreate(
        company_ticker=company_ticker,
        document_type=document_type,
        year=year,
        is_public=is_public,
    )
