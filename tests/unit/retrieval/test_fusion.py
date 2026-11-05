import pytest

from financial_assistant.retrieval.fusion import reciprocal_rank_fusion

pytestmark = pytest.mark.unit


def test_scores_sum_reciprocal_ranks_across_lists():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1]], k=60)
    assert fused == [
        (1, pytest.approx(1 / 61 + 1 / 62)),
        (3, pytest.approx(1 / 63 + 1 / 61)),
        (2, pytest.approx(1 / 62)),
    ]


def test_an_item_ranked_in_both_lists_beats_a_single_first_place():
    fused = reciprocal_rank_fusion([[10, 20], [30, 20]], k=60)
    assert fused[0][0] == 20


def test_one_empty_list_keeps_the_other_ranking():
    assert [item for item, _ in reciprocal_rank_fusion([[], [5, 4, 9]], k=60)] == [
        5,
        4,
        9,
    ]


def test_both_empty_lists_fuse_to_nothing():
    assert reciprocal_rank_fusion([[], []], k=60) == []


def test_ties_keep_the_order_of_first_appearance():
    # 7 and 8 both score 1/61 + 1/62; 7 appears first.
    fused = reciprocal_rank_fusion([[7, 8], [8, 7]], k=60)
    assert [item for item, _ in fused] == [7, 8]
    assert fused[0][1] == pytest.approx(fused[1][1])
