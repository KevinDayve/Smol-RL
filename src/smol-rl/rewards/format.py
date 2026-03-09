import re
from .base import BaseReward

class FormatReward(BaseReward):
    """
    See if the response is in the correct structure.
    """

    def score(self, completion: str, reference: str) -> float:
        reward = 0.0
        think_match = re.search("<think>(.+?)</think>", completion, re.DOTALL)
        answer_match = re.search("<answer>(.+?)</answer", completion, re.DOTALL)

        if think_match and think_match.group(1).strip():
            reward += 0.1
        if answer_match and answer_match.group(1).strip():
            reward += 0.1
        if "<think>" in completion and "</think>" not in completion:
            reward -= 0.1

        return reward