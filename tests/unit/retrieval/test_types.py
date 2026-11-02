import pytest

from financial_assistant.models.document import DocumentType
from financial_assistant.retrieval.types import (
    ChunkSource,
    CommunitySource,
    RetrievalScope,
    source_key,
)

pytestmark = pytest.mark.unit


def test_narrowing_an_unfiltered_scope_adds_the_filters():
    scope = RetrievalScope(user_id=1).narrowed(
        company_ticker="NVDA", document_type=DocumentType.ANNUAL_REPORT, year=2025
    )
    assert scope == RetrievalScope(
        user_id=1,
        company_ticker="NVDA",
        document_type=DocumentType.ANNUAL_REPORT,
        year=2025,
    )
    assert not scope.is_empty


def test_narrowing_keeps_the_thread_filters():
    thread = RetrievalScope(user_id=1, document_ids=(3, 4), company_ticker="NVDA")
    assert thread.narrowed(year=2024) == RetrievalScope(
        user_id=1, document_ids=(3, 4), company_ticker="NVDA", year=2024
    )
    assert thread.narrowed() == thread


def test_a_matching_ticker_is_compared_case_insensitively():
    thread = RetrievalScope(user_id=1, company_ticker="NVDA")
    assert thread.narrowed(company_ticker="nvda") == thread


@pytest.mark.parametrize(
    "request_",
    [
        {"company_ticker": "TSLA"},
        {"document_type": DocumentType.QUARTERLY_REPORT},
        {"year": 2023},
    ],
)
def test_a_conflicting_filter_never_widens_and_yields_an_empty_scope(request_):
    current = RetrievalScope(
        user_id=1,
        company_ticker="NVDA",
        document_type=DocumentType.ANNUAL_REPORT,
        year=2025,
    )
    narrowed = current.narrowed(**request_)
    assert narrowed.is_empty
    assert narrowed.company_ticker == "NVDA"


def test_an_empty_scope_stays_empty():
    empty = RetrievalScope(user_id=1, document_ids=())
    assert empty.is_empty
    assert empty.narrowed(company_ticker="NVDA").is_empty


def test_source_keys_separate_kinds_and_ids():
    chunk = ChunkSource(
        parent_chunk_id=7,
        document_id=1,
        filename="a.pdf",
        company_ticker="A",
        document_type=DocumentType.ANNUAL_REPORT,
        year=2025,
        content="text",
        score=0.03,
    )
    community = CommunitySource(
        community_id=7,
        document_id=1,
        filename="a.pdf",
        company_ticker="A",
        title="t",
        summary="s",
        similarity=0.9,
    )
    assert source_key(chunk) == ("chunk", 7)
    assert source_key(community) == ("community", 7)
