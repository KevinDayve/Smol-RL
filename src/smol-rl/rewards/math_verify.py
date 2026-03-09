import re
from typing import Optional, List
from math_verify import parse, verify
from math_verify.extraction import LatexExtractionConfig, ExprExtractionConfig
from .base import BaseReward

class MathVerifyReward(BaseReward):
    """
    Math correctness using math-verify.
    """
    def __init__(self, extraction_configs: Optional[List] = None) -> None:
        """
        Args:
            extraction_configs: List of math verify extraction configs. Defaults to LatexExtraction and ExprExtraction...
        """
        super().__init__()
        self.extraction_configs = extraction_configs if extraction_configs is not None else [LatexExtractionConfig(), ExprExtractionConfig()]
    def _extract_answer(self, completion: str) -> str:
        """
        Using regex, attempt to pull the following out:
        1). <answer>..</answer> tags.
        2). \\boxed{...}
        3). If both fail, raw.
        """
        match = re.search(r"<answer>(.*?)</answer>", completion, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        
        index = completion.rfind("\\{boxed")
        if index != -1:
            start = index + len("\\{boxed")
            depth = 1
            for i in range(start, len(completion)):
                if completion[i] == "{":
                    depth += 1
                elif completion[i] == "}":
                    depth -= 1
                    if depth == 0:
                        return completion[start:i].strip()
        
        # raw compltion
        return completion.strip()
    def score(self, completion: str, reference: str, incorr_penalty: float = 0.0) -> float:
        try:
            extracted = self._extract_answer(completion=completion)
            parsed_answer = parse(extracted, extraction_config=self.extraction_configs)
            parsed_gold = parse(reference, extraction_config=self.extraction_configs)
            return 1.0 if parsed_answer==parsed_gold else incorr_penalty
        except Exception:
            return incorr_penalty