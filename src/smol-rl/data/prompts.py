from typing import Optional, List, Dict
from datasets import load_dataset
from ..rewards.math_verify import MathVerifyReward

def read_prompts(dataset_name: str, max_rows: Optional[int] = None) -> List[Dict]:
    """
    Loads a maths Q/A dataset and extracts the ground truth answers.
    Returns a list of dicts with "problem" and "answer" keys.
    """
    dataset = load_dataset(dataset_name)
    rows = dataset.to_list()
    reward = MathVerifyReward()

    clean = []
    for row in rows:
        answer = reward._extract_answer(row.get("solution", ""))
        if answer and answer != row.get("solution", "").strip():
            row['answer'] = answer
            clean.append(row)
    
    if max_rows is not None:
        clean = clean[:max_rows]
    return clean