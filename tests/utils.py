from typing import Protocol

from financial_assistant.models import Document, User
from financial_assistant.models.document import DocumentStatus, DocumentType


class UserFactory(Protocol):
    async def __call__(self, email: str, password: str = "securepassword") -> User: ...


class TokenFactory(Protocol):
    def __call__(self, user: User) -> str: ...


class DocumentFactory(Protocol):
    async def __call__(
        self,
        owner_id: int | None = None,
        company_ticker: str = "AAPL",
        document_type: DocumentType = DocumentType.ANNUAL_REPORT,
        year: int = 2023,
        filename: str = "test.pdf",
        status: DocumentStatus = DocumentStatus.COMPLETED,
    ) -> Document: ...
