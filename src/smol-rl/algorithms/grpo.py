from typing import Optional, Dict
import torch
from .base import BaseLoss, approximateKLDivergence, masked_mean

class GRPOLoss(BaseLoss):
    """
    Group-relative policy optimisaiton.
    Core functionality:
        - Symmetric ratio clipping [1 - epsilon, 1 + epsilon]
        - Group relative advantage normalisation
        - Additive KL penalty. (-Clipped + Beta * KL)
    """
    def __init__(self, epsilon: float = 0.2, kl_beta: float = 0.04) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.kl_beta = kl_beta
    
    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Compute per token KL between current and reference policy.
        kl = approximateKLDivergence(
            log_probabilities=log_probs,
            reference_log_probabilities=ref_log_probs,
            action_mask=action_mask
        )
        # Then, we compute per-token importance sampling ratio between current and behaviour policy.
        ratio = torch.exp(log_probs - old_log_probs)
        # Clamp symmetrically.
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        # clip fraction (for logging purposes, if consistently near zero, the clip is too wide)
        clip_frac = (ratio != clipped_ratio).float().mean()
        # Compute the clipped surrogate.
        surrogate = torch.min(ratio * advantages, clipped_ratio * advantages)
        # per token loss
        loss = -surrogate + self.kl_beta * kl
        # aggreagate with masked mean and then mean over batch.
        loss = masked_mean(loss, mask=action_mask, dim=-1).mean()
        kl = masked_mean(kl, mask=action_mask, dim=-1).mean()
        return {
            "loss": loss,
            "kl": kl,
            "clip_fraction": clip_frac,
        }
