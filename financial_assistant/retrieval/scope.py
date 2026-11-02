"""The only place that encodes which documents a search may read."""

from sqlalchemy import Select, func, or_, select

from ..models import Document
from ..models.document import DocumentStatus
from .types import RetrievalScope


def accessible_documents(scope: RetrievalScope) -> Select[tuple[int]]:
    """
    Ids of the completed documents in the scope: public (owner_id IS NULL) or owned
    by the user. A deleted account's documents keep their owner_id (a tombstoned
    user who can't sign in), so they match nobody. Every retriever filters on this.
    """
    query = select(Document.id).where(
        Document.status == DocumentStatus.COMPLETED,
        or_(Document.owner_id.is_(None), Document.owner_id == scope.user_id),
    )
    if scope.document_ids is not None:
        query = query.where(Document.id.in_(scope.document_ids))
    if scope.company_ticker is not None:
        query = query.where(
            func.upper(Document.company_ticker) == scope.company_ticker.upper()
        )
    if scope.document_type is not None:
        query = query.where(Document.document_type == scope.document_type)
    if scope.year is not None:
        query = query.where(Document.year == scope.year)
    return query
