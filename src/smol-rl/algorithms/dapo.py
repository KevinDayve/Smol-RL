from typing import Optional, Dict
import torch
from .base import BaseLoss, masked_mean, approximateKLDivergence

class DAPOLoss(BaseLoss):
    """
    Decoupled clip and dynamic sampling policy optimisation.
    Core functionality:
        - Asymmetric clipping [1 - epsilon_low, 1 + epsilon_high] where eps_high > epsilon_low. This biases the update towards exploitation of positive signal.
        - Explicit token level entropy bonus.
        - NO KL penalty.
    """
    def __init__(self, eps_low: float = 0.2, eps_high: float = 0.28, entropy_coef: float = 0.001) -> None:
        super().__init__()
        self.eps_low = eps_low
        self.eps_high = eps_high
        self.entropy_coef = entropy_coef
    def forward(self, log_probs: torch.Tensor, old_log_probs: torch.Tensor, ref_log_probs: torch.Tensor, advantages: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # compute per-token importance ration.
        ratio = torch.exp(log_probs - old_log_probs)
        # for logging purposes, we compute kl divergence (altho it won't be added to final loss)
        kl = approximateKLDivergence(
            log_probabilities=log_probs,
            reference_log_probabilities=ref_log_probs,
            action_mask=action_mask,
        )

        # asymmetric clipping.
        clipped_ratio = torch.clamp(ratio, 1 - self.eps_low, 1 + self.eps_high)
        # clip fraction
        clip_frac = (ratio != clipped_ratio).float().mean()

        # surrogate objective.
        surrogate = torch.min(ratio * advantages, clipped_ratio * advantages)
        # entropy bonus of the sampled token.
        entropy = -log_probs
        # per token loss
        loss = -surrogate - self.entropy_coef * entropy
        # aggregate with masked mean then mean over batch.
        loss = masked_mean(loss, mask=action_mask, dim=-1).mean()
        kl = masked_mean(kl, mask=action_mask, dim=-1).mean()
        return {
            "loss": loss,
            "kl": kl, # just for logging purposes,
            "clip_fraction": clip_frac,
        }