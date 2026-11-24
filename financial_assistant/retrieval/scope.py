"""The only place that encodes which documents a search may read."""

from sqlalchemy import ColumnElement, Select, func, or_, select

from ..models import Document
from ..models.document import DocumentStatus
from .types import RetrievalScope


def readable_documents(user_id: int) -> ColumnElement[bool]:
    """
    Public (owner_id IS NULL) or owned by the user, whatever the status. A deleted
    account's documents keep their owner_id (a tombstoned user who can't sign in),
    so they match nobody.
    """
    return or_(Document.owner_id.is_(None), Document.owner_id == user_id)


def accessible_documents(scope: RetrievalScope) -> Select[tuple[int]]:
    """
    Ids of the completed documents in the scope that the user may read. Every
    retriever filters on this.
    """
    query = select(Document.id).where(
        Document.status == DocumentStatus.COMPLETED,
        readable_documents(scope.user_id),
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
