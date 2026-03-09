from dataclasses import dataclass, fields
from typing import Optional, Self, List
import torch
import torch.nn.functional as Functional

def pad_with_zeros(sequences: List[torch.Tensor], padding_side: str = "left") -> torch.Tensor:
    """
    Pad a list of variable length sequences to the same length.
    Args:
        sequences: List of (T_i, ) tensors.
        padding_side: "left" for causal modelling and right for encoders. Defaults to left.
    Returns:
        (N, T_max) tensor.
    """
    assert padding_side in ("left", "right")
    maxLen = max(seq.size(-1) for seq in sequences)
    padded: List = []
    for sequence in sequences:
        pad_length = maxLen - sequence.size(-1)
        pad = (pad_length, 0) if padding_side == "left" else (0, pad_length)
        padded.append(Functional.pad(sequence, pad))
    return torch.stack(padded, dim=0)

@dataclass
class Experience:
    """
    One batch of RL "experience" from a rollout step.
    All tensors are of shape (B, T) or (B, 1) unless specially mentioned.
    """
    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    reference_log_probs: torch.Tensor
    action_mask: torch.Tensor
    returns: Optional[torch.Tensor] = None
    advantages: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    kl: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> Self:
        members = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, torch.Tensor):
                v = v.to(device=device)
            members[f.name] = v
        return Experience(**members)
    
def split_batch_experience(experience: Experience) -> List[Experience]:
    bsz = experience.sequences.size(0)
    batch_data = [{} for _ in range(bsz)]
    for f in fields(experience):
        value = getattr(experience, f.name)
        if value is None:
            vals = [None] * bsz
        else:
            vals = torch.unbind(value)
        for i, v in enumerate(vals):
            batch_data[i][f.name] = v
    return [Experience(**data) for data in batch_data]

def join_experience_batch(items: List[Experience]) -> Experience:
    """
    Collate a list of single-sample experiences back intoa  batch with left padding.
    """
    batch_data = {}
    for f in fields(items[0]):
        values = [getattr(item, f.name) for item in items]
        if all(v is not None for v in values):
            batch_data[f.name] = pad_with_zeros(values, padding_side="left")
        else:
            batch_data[f.name] = None
    return Experience(**batch_data)