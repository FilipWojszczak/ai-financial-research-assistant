from typing import Protocol

from google.genai.errors import ClientError, ServerError
from langchain_google_genai._common import GoogleGenerativeAIError
from sqlalchemy.engine import make_url

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


def provider_error(code: int) -> GoogleGenerativeAIError:
    """A Gemini HTTP error wrapped by LangChain the way the SDK integration does it."""
    error_type = ClientError if code < 500 else ServerError
    body = {"error": {"code": code, "message": "test", "status": "TEST"}}
    try:
        try:
            raise error_type(code, body)
        except error_type as sdk_error:
            raise GoogleGenerativeAIError(f"Error ({code})") from sdk_error
    except GoogleGenerativeAIError as wrapped:
        return wrapped


def derive_test_database_url(database_url: str) -> str:
    """Return the same URL pointing at `<database>_test`.

    A URL whose database already ends in `_test` is returned unchanged (idempotent).
    """
    url = make_url(database_url)
    if not url.database:
        raise ValueError("The database URL has no database name.")
    if not url.database.endswith("_test"):
        url = url.set(database=url.database + "_test")
    return url.render_as_string(hide_password=False)
