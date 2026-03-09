from .base import BaseReward


class CompositeReward:
    """
    Weighted combination of arbitrary reward scorers.
    """

    def __init__(self, scorers: list[tuple[BaseReward, float]]) -> None:
        self.scorers = scorers

    def score(self, completion: str, reference: str) -> float:
        total = 0.0
        for scorer, weight in self.scorers:
            total += weight * scorer.score(completion, reference)
        return total