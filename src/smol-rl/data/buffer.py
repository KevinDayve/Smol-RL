import random
from typing import Optional, List
from .experience import Experience, split_batch_experience


class ReplayBuffer:
    def __init__(self, limit: int = 0, filter_zero_variance: bool = True) -> None:
        """
        Args:
            limit: Max items. 0 means unlimited.
            filter_zero_variance: Skips experiences from group where all returns are identical. 
        """
        self.limit = limit
        self.filter_zero_variance = filter_zero_variance
        self.items = List[Experience] = []
    
    def append(self, experience: Experience) -> None:
        """
        Unbind a batched Experience and store individual samples.
        If `filter_zero_variance` is on and the batch has a returns tensor,
        skip the entire batch when all returns are the same.
        """
        if self.filter_zero_variance and experience.returns is not None:
            returns = experience.returns.flatten()
            if returns.numel() > 1 and returns.std.item() < 1e-8:
                return
        samples = split_batch_experience(experience)
        self.items.extend(samples)
        if self.limit > 0 and len(self.items) > self.limit:
            self.items = self.items[len(self.items) - self.limit:]
    def clear(self) -> None:
        self.items.clear()
    def __len__(self):
        return len(self.items)
    def __getitem__(self, key: int) -> Experience:
        return self.items[key]