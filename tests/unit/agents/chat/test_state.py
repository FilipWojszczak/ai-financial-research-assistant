from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.types import Overwrite

from financial_assistant.agents.chat.models import RetrievalGrade
from financial_assistant.agents.chat.state import (
    chat_serializer,
    merge_dicts,
    merge_sources,
)
from tests.utils import chunk_source, community_source

pytestmark = pytest.mark.unit


def test_sources_are_numbered_in_order_of_first_appearance():
    first, second = chunk_source(1), chunk_source(2)
    merged = merge_sources([], [first, second])
    assert merged == [{"ref": 1, "source": first}, {"ref": 2, "source": second}]


def test_duplicates_are_dropped_and_refs_stay_stable_across_rounds():
    chunk, community = chunk_source(1), community_source(1)
    round_one = merge_sources([], [chunk, community])
    round_two = merge_sources(round_one, [chunk, chunk_source(2), chunk])
    assert [(item["ref"], item["source"]) for item in round_two] == [
        (1, chunk),
        (2, community),
        (3, chunk_source(2)),
    ]


def test_merging_does_not_mutate_the_current_value():
    current = merge_sources([], [chunk_source(1)])
    merge_sources(current, [chunk_source(2)])
    assert len(current) == 1


def test_overwrite_bypasses_the_reducer_to_reset_sources():
    channel: Any = BinaryOperatorAggregate(list, cast(Any, merge_sources))
    channel.update([[chunk_source(1)]])
    assert len(channel.get()) == 1
    channel.update([Overwrite([])])
    assert channel.get() == []


def test_merge_dicts_keeps_earlier_keys():
    assert merge_dicts({"a": [1]}, {"b": [2]}) == {"a": [1], "b": [2]}


def test_checkpoints_rebuild_every_stored_type():
    # With an explicit allowlist, an unlisted type would come back as plain data.
    serde = chat_serializer()
    state = {
        "sources": merge_sources([], [chunk_source(1), community_source(1)]),
        "grade": RetrievalGrade(sufficient=False, missing="the 2024 figure"),
        "messages": [HumanMessage("q", id="t:q"), AIMessage("a", id="t")],
    }
    assert serde.loads_typed(serde.dumps_typed(state)) == state
