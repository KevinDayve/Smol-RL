import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from ..config import Config
from ..algorithms.base import approximateKLDivergence, masked_mean
from ..algorithms.grpo import GRPOLoss
from ..algorithms.dapo import DAPOLoss
from ..data.experience import Experience, join_experience_batch
from ..data.buffer import ReplayBuffer
from ..rollout.vllm_rollout import VLLMRolloutManager, format_prompt
from ..rewards.math_verify import MathVerifyReward
from ..rewards.format import FormatReward
from ..rewards.composite import CompositeReward


logger = logging.getLogger(__name__)


LOSS_REGISTRY = {
    "grpo": GRPOLoss,
    "dapo": DAPOLoss,
}


def load_model(
    model_name_or_path: str,
    trust_remote_code: bool = False,
    bf16: bool = True,
    device_map=None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model and tokeniser. Sets pad_token to eos_token."""
    tokeniser = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    tokeniser.pad_token = tokeniser.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        device_map=device_map,
    )
    return model, tokeniser


def sequence_log_probs_from_logits(
    logits: torch.Tensor,
    output_ids: torch.Tensor,
) -> torch.Tensor:
    """Convert (B, T, V) logits to (B, T) per-token log probs of the actual tokens."""
    return -F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        output_ids.reshape(-1),
        reduction="none",
    ).reshape(output_ids.shape)


def compute_log_probs(
    model: PreTrainedModel,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    chunk_size: int = 4,
) -> torch.Tensor:
    """
    Forward pass in chunks to avoid OOM.

    Returns (B, T-1) log probs — shifted by one to align
    with the action_mask convention.
    """
    all_log_probs = []
    for i in range(0, sequence_ids.size(0), chunk_size):
        chunk_ids = sequence_ids[i : i + chunk_size]
        chunk_mask = attention_mask[i : i + chunk_size]

        position_ids = chunk_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(chunk_mask == 0, 1)

        output = model(
            input_ids=chunk_ids,
            attention_mask=chunk_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        logits = output["logits"]
        log_probs = sequence_log_probs_from_logits(
            logits[:, :-1].to(torch.float32),
            chunk_ids[:, 1:],
        )
        all_log_probs.append(log_probs)
        del output, logits

    return torch.cat(all_log_probs, dim=0)


def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Group-level normalisation: (r - mean) / (std + eps)."""
    return (returns - returns.mean()) / (returns.std() + eps)


def build_loss(cfg: Config) -> torch.nn.Module:
    """Construct the loss from config."""
    loss_cfg = cfg.loss
    cls = LOSS_REGISTRY[loss_cfg.name]

    if loss_cfg.name == "grpo":
        return cls(epsilon=loss_cfg.clip_eps, kl_beta=loss_cfg.kl_beta)
    elif loss_cfg.name == "dapo":
        return cls(eps_low=loss_cfg.eps_low, eps_high=loss_cfg.eps_high, entropy_coef=loss_cfg.entropy_coef)


def build_reward_fn(cfg: Config) -> CompositeReward:
    """Default reward composition."""
    return CompositeReward([
        (MathVerifyReward(), 1.0),
        (FormatReward(), 0.5),
    ])


class Trainer:
    """
    Main training loop.

    Lifecycle:
        1. Load models (policy, reference, vLLM engine).
        2. Load dataset.
        3. For each batch of prompts:
           a. Generate grouped rollouts via vLLM.
           b. Compute log probs under policy and reference.
           c. Compute advantages.
           d. Store in replay buffer.
           e. Train for epochs_per_step epochs over the buffer.
           f. Sync weights to vLLM periodically.
           g. Checkpoint periodically.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda", cfg.training.device_index)

        self.reference_model, _ = load_model(
            cfg.model.name,
            trust_remote_code=cfg.model.trust_remote_code,
            bf16=cfg.model.bf16,
            device_map=self.device,
        )
        self.model, self.tokeniser = load_model(
            cfg.model.name,
            trust_remote_code=cfg.model.trust_remote_code,
            bf16=cfg.model.bf16,
            device_map=self.device,
        )

        self.reference_model.eval()
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        self.optimiser = optim.AdamW(
            self.model.parameters(),
            lr=float(cfg.training.lr),
        )
        self.objective = build_loss(cfg)
        self.reward_fn = build_reward_fn(cfg)
        self.buffer = ReplayBuffer(filter_zero_variance=True)
        self.rollout_manager = VLLMRolloutManager(
            model_name_or_path=cfg.model.name,
            rollout_cfg=cfg.rollout,
        )

        self.checkpoint_dir: Optional[Path] = None
        if cfg.training.checkpoint_path:
            self.checkpoint_dir = Path(cfg.training.checkpoint_path) / cfg.wandb.run_name

    def _compute_experience(
        self,
        sequence_ids: torch.Tensor,
        action_mask: torch.Tensor,
        returns: torch.Tensor,
    ) -> Experience:
        """
        Given raw rollout data on device, compute log probs under
        both policies, advantages, and return a complete Experience.
        """
        attention_mask = sequence_ids != self.tokeniser.eos_token_id

        log_probs = compute_log_probs(self.model, sequence_ids, attention_mask)
        ref_log_probs = compute_log_probs(self.reference_model, sequence_ids, attention_mask)
        kl = approximateKLDivergence(log_probs, ref_log_probs, action_mask)
        advantages = group_advantages(returns)

        return Experience(
            sequences=sequence_ids,
            action_log_probs=log_probs,
            ref_log_probs=ref_log_probs,
            action_mask=action_mask,
            returns=returns,
            advantages=advantages,
            attention_mask=attention_mask,
            kl=kl,
        )

    def _train_step(self, epoch: int, experience_sampler: DataLoader) -> dict:
        """
        One epoch of training over the replay buffer.

        This is where gradient accumulation happens:
            - Forward pass on each micro-batch.
            - Scale loss by 1/grad_acc_steps.
            - Backward.
            - Every grad_acc_steps micro-batches: clip, step, zero_grad.

        Returns:
            Dict of metrics from the last optimiser step
            (loss, kl, grad_norm) for logging.
        """
        self.model.train()
        self.optimiser.zero_grad()

        grad_acc = self.cfg.training.grad_acc_steps
        metrics = {}

        for micro_step, exp in enumerate(experience_sampler):
            exp = exp.to(self.device)

            log_probs = compute_log_probs(
                self.model,
                sequence_ids=exp.sequences,
                attention_mask=exp.attention_mask,
            )

            result = self.objective(
                log_probs=log_probs,
                old_log_probs=exp.action_log_probs,
                ref_log_probs=exp.ref_log_probs,
                advantages=exp.advantages,
                action_mask=exp.action_mask,
            )

            loss = result["loss"] / grad_acc

            if not loss.isfinite():
                logger.warning(f"Non-finite loss at micro_step {micro_step}, skipping.")
                continue

            loss.backward()

            if (micro_step + 1) % grad_acc == 0:
                grad_norm = clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.cfg.training.max_norm,
                )
                self.optimiser.step()
                self.optimiser.zero_grad()

                metrics = {
                    "loss": loss.item() * grad_acc,  # unscaled for logging
                    "kl": result["kl"].item(),
                    "grad_norm": grad_norm.item(),
                }

        return metrics

    def train(self, prompts: list[dict]) -> None:
        """
        Main loop. Takes the full dataset as a list of dicts
        with 'problem' and 'answer' keys.

        This is the method you call from the CLI entry point.

        Steps per iteration:
            1. Sample a batch of prompts.
            2. Generate rollouts.
            3. Compute log probs and advantages.
            4. Fill buffer.
            5. Train.
            6. Sync weights / checkpoint as needed.
            7. Log.
        """
        prompt_loader = DataLoader(
            prompts,
            batch_size=self.cfg.rollout.rollouts_per_step,
            shuffle=True,
            drop_last=True,
        )

        total_steps = len(prompt_loader)
        logger.info(
            f"Training: {len(prompts)} prompts, {total_steps} steps, "
            f"group_size={self.cfg.rollout.group_size}"
        )

        for step, batch in enumerate(prompt_loader):
            self.buffer.clear()
            questions = batch["problem"]
            answers = batch["answer"]

            rollout_results = self.rollout_manager.generate(
                tokeniser=self.tokeniser,
                questions=questions,
                references=answers,
                reward_function=self.reward_fn,
            )

            all_rewards = []
            self.model.eval()
            with torch.no_grad():
                for experience, completions in rollout_results:
                    exp_device = Experience(
                        sequences=experience.sequences.to(self.device),
                        action_log_probs=None,
                        ref_log_probs=None,
                        action_mask=experience.action_mask.to(self.device),
                        returns=experience.returns.to(self.device),
                    )

                    full_exp = self._compute_experience(
                        sequence_ids=exp_device.sequences,
                        action_mask=exp_device.action_mask,
                        returns=exp_device.returns,
                    )

                    all_rewards.extend(experience.returns.flatten().tolist())
                    self.buffer.append(full_exp.to(torch.device("cpu")))

            torch.cuda.empty_cache()

            if len(self.buffer) < self.cfg.training.train_batch_size:
                logger.warning(f"Step {step}: buffer too small ({len(self.buffer)}), skipping training.")
                continue

            experience_sampler = DataLoader(
                self.buffer,
                batch_size=self.cfg.training.train_batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=join_experience_batch,
            )

            metrics = {}
            for epoch in range(self.cfg.training.epochs_per_step):
                metrics = self._train_step(epoch, experience_sampler)

            reward_mean = sum(all_rewards) / len(all_rewards)
            accuracy = sum(1 for r in all_rewards if r >= 0.8) / len(all_rewards)
            logger.info(
                f"Step {step}: reward={reward_mean:.3f}, acc={accuracy:.1%}, "
                f"loss={metrics.get('loss', 0):.4f}, kl={metrics.get('kl', 0):.4f}"
            )

            if (step + 1) % self.cfg.rollout.sync_interval == 0:
                self.rollout_manager.sync_weights(self.model, self.tokeniser)

            if (
                self.checkpoint_dir
                and self.cfg.training.checkpoint_interval
                and (step + 1) % self.cfg.training.checkpoint_interval == 0
            ):
                save_path = self.checkpoint_dir / f"step_{step + 1}"
                save_path.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(save_path)
                self.tokeniser.save_pretrained(save_path)
                logger.info(f"Checkpoint saved: {save_path}")

        if self.checkpoint_dir:
            save_path = self.checkpoint_dir / "final"
            save_path.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(save_path)
            self.tokeniser.save_pretrained(save_path)
            logger.info(f"Final model saved: {save_path}")

        self.rollout_manager.shutdown()