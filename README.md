# tinyrl

A minimal, correct LLM-RL training library.

## What's different

- **Typed config** — nested dataclasses with validation, not raw dicts.
- **Pluggable algorithms** — GRPO and DAPO behind a common `BaseLoss` interface. Adding a new algorithm = one file.
- **Pluggable rewards** — `BaseReward` interface with a `CompositeReward` combinator. Uses [math-verify](https://github.com/huggingface/Math-Verify) instead of hand-rolled LaTeX normalisation.
- **Zero-variance filtering** — replay buffer drops groups where all rewards are identical (zero gradient signal under group normalisation).
- **AdamW** — not Adam. Weight decay matters for fine-tuning.
- **Proper gradient accumulation** — handles incomplete final batches, logs unscaled loss.
- **Clean separation** — rollout, training, rewards, and algorithms in separate modules with explicit interfaces.

## Structure
```
tinyrl/
├── __init__.py
├── cli.py                  # entry point
├── config.py               # typed dataclass config
├── algorithms/
│   ├── base.py             # BaseLoss + shared utilities
│   ├── grpo.py             # Group Relative Policy Optimisation
│   └── dapo.py             # Decoupled clip + entropy bonus
├── rewards/
│   ├── base.py             # BaseReward interface
│   ├── math_verify.py      # math-verify backed scorer
│   ├── format.py           # structural format scorer
│   └── composite.py        # weighted reward combinator
├── data/
│   ├── experience.py       # Experience dataclass + padding
│   ├── buffer.py           # ReplayBuffer with filtering
│   └── prompts.py          # dataset loading
├── rollout/
│   └── vllm_rollout.py     # vLLM generation + weight sync
├── training/
│   └── trainer.py          # main training loop
└── utils/
    └── seed.py             # reproducibility
```

## Setup
```bash
uv sync
```

## Usage
```bash
tinyrl-train --config configs/grpo_qwen3.yaml
```

Or directly:
```bash
python -m tinyrl.cli --config configs/grpo_qwen3.yaml
```

## Config

All hyperparameters live in YAML. See `configs/grpo_qwen3.yaml` for the full set.

Key knobs:
- `loss.name` — `grpo` or `dapo`
- `rollout.group_size` — completions per question
- `training.lr` — learning rate
- `training.grad_acc_steps` — gradient accumulation
- `rollout.sync_interval` — how often to push weights to vLLM

## Adding a new algorithm

1. Create `algorithms/your_algo.py` inheriting from `BaseLoss`.
2. Implement `forward()` returning `dict[str, Tensor]` with at least `"loss"` and `"kl"`.
3. Add it to `LOSS_REGISTRY` in `training/trainer.py`.
4. Add its config fields to `LossConfig` in `config.py`.

## Adding a new reward

1. Create `rewards/your_reward.py` inheriting from `BaseReward`.
2. Implement `score(completion, reference) -> float`.
3. Add it to `build_reward_fn()` in `training/trainer.py`.