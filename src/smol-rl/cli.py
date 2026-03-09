import argparse
import logging

import wandb

from .config import Config
from .data.prompts import read_prompts
from .training.train import Trainer
from .utils.seed import init_rng


def main() -> None:
    parser = argparse.ArgumentParser(description="smol-rl — LLM-RL training")
    parser.add_argument("--config", type=str, default="configs/grpo_qwen3.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = Config.from_yaml(args.config)
    init_rng(cfg.training.seed)

    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            config=vars(cfg),
        )

    prompts = read_prompts(
        dataset_name=cfg.dataset.name,
        max_rows=cfg.dataset.max_rows,
    )

    trainer = Trainer(cfg)
    trainer.train(prompts)

    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()