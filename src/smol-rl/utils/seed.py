import random
import torch


def init_rng(seed: int) -> torch.Generator:
    """Seed everything for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch.manual_seed(seed)