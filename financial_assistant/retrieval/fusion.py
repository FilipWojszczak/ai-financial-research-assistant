from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], *, k: int
) -> list[tuple[int, float]]:
    """
    Fuse rankings (best first) with Reciprocal Rank Fusion: score = Σ 1 / (k + rank),
    1-based ranks, equal weights. It favours items ranked well in several lists and
    needs no normalization between unbounded BM25 scores and cosine distances.
    Ties keep the order of first appearance, so the result is deterministic.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1 / (k + rank)
    # sorted() is stable and dicts keep insertion (first appearance) order.
    return sorted(scores.items(), key=lambda item: -item[1])
