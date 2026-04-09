from pydantic import BaseModel, ConfigDict, Field

from ..models.document import DocumentType


class DocumentCreate(BaseModel):
    filename: str = Field(description="The name of the file to be uploaded")
    company_ticker: str = Field(
        description="The stock ticker symbol of the company associated with the "
        "document"
    )
    document_type: DocumentType = Field(description="The type of the document")
    year: int = Field(description="The year the document was published")
    is_public: bool = Field(
        description="Indicates whether the document is public or belongs to a specific "
        "user"
    )

    model_config = ConfigDict(extra="forbid")


class DocumentRead(BaseModel):
    id: int
    filename: str
    company_ticker: str
    document_type: DocumentType
    year: int
    owner_id: int | None

    model_config = ConfigDict(from_attributes=True)
