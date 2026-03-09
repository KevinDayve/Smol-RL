from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal
import yaml

@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen3-1.7B"
    trust_remote_code: bool = True
    bf16: bool = True

@dataclass
class DatasetConfig:
    name: str = "qwedsacf/competition_math"
    max_rows: Optional[int] = None

@dataclass
class TrainingConfig:
    seed: int = 42
    device_index: int = 0
    lr: float = 1e-5
    train_batch_size: int = 4
    grad_acc_steps: int = 4
    epochs_per_step: int = 1
    max_norm: float = 1.0
    checkpoint_path: Optional[str] = "./output"
    checkpoint_interval: int = 20

@dataclass
class LossConfig:
    name: Literal['grpo', 'dapo'] = 'grpo'
    # GRPO specific
    clip_eps: float = 0.2
    kl_beta: float = 0.04
    # DAPO specific
    eps_low: float = 0.2
    eps_high: float = 0.28
    entropy_coef: float = 0.001


@dataclass
class RolloutConfig:
    group_size: int = 12
    rollouts_per_step: int = 32
    max_length: int = 2048
    top_p: float = 1.0
    temperature: float = 1.0
    gpu_memory_utilization: float = 0.3
    sync_interval: int = 2


@dataclass
class WandbConfig:
    project: str = "RL"
    run_name: str = "grpo"
    enabled: bool = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """
        Load from YAML, validate and return typed config.
        """
        with open(path, "r") as yamlfile:
            raw = yaml.safe_load(yamlfile)
        return cls(
        model=ModelConfig(**raw.get("model", {})),
        dataset=DatasetConfig(**raw.get("dataset", {})),
        training=TrainingConfig(**raw.get("training", {})),
        loss=LossConfig(**raw.get("loss", {})),
        rollout=RolloutConfig(**raw.get("rollout", {})),
        wandb=WandbConfig(**raw.get("wandb", {})),
    )