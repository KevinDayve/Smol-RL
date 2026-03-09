import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict


def approximateKLDivergence(
        log_probabilities: torch.Tensor,
        reference_log_probabilities: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Schulman's second order KL approximation: exp(log_ratio) - log_ration - 1.
    Arguments:
        log_probabilities: Log probabilities of the current policy.
        reference_log_probabilities: Log probabilities of the reference policy.
        action_mask: Optional mask to apply to the KL divergence calculation.
    Returns:
        The approximate KL divergence between the current policy and the reference policy.
    """
    log_ratio = (log_probabilities - reference_log_probabilities)
    if action_mask is not None:
        log_ratio = log_ratio * action_mask
    return torch.exp(log_ratio) - log_ratio - 1

def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor] = None, dim: Optional[int] = None) -> torch.Tensor:
    """
    Mean over non-masked positions. Returns the global mean if mask is not present 
    """
    dim = dim if dim is not None else -1
    if mask is not None:
        masked_tensor = tensor * mask
        mean = masked_tensor.sum(dim=dim) / (mask.sum(dim=dim) + 1e-8)
        return mean
    else:
        return tensor.mean(dim=dim)


class BaseLoss(ABC, nn.Module):
    """
    Abstract base class for loss functions in TinyRL.
    """
    
    @abstractmethod
    def forward(self, log_probabilities: torch.Tensor, old_log_probabilities: torch.Tensor, reference_log_probabilities: torch.Tensor, advantages: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Arguments:
            log_probabilities: Current policy log probs (B, T)
            old_log_probabilities: Behaviour policy log probs (B, T)
            reference_log_probabilities: Reference policy log probs (B, T).
            advantages: Per token or sequence advantages.
            action_mask: Boolean mask over generated tokens.
        Returns:
            Dict with "loss" and "kl" key at the mminum.
        """
        pass