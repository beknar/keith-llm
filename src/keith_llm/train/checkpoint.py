"""Checkpoint save/load including optimizer and RNG states for exact resume."""

from __future__ import annotations

import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from keith_llm.config import ModelConfig, TrainConfig
from keith_llm.model import Transformer


def save_checkpoint(
    path: str | Path,
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    step: int,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    np_rng: np.random.Generator,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "model_cfg": asdict(model_cfg),
        "train_cfg": asdict(train_cfg),
        "rng": {
            "python": random.getstate(),
            "numpy": np_rng.bit_generator.state,
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic: never leaves a torn latest.pt


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    # weights_only=False: checkpoints are produced by this project and carry
    # RNG state objects, not just tensors. Never load untrusted checkpoints.
    return torch.load(path, map_location=map_location, weights_only=False)


def restore_rng(rng_state: dict[str, Any], np_rng: np.random.Generator) -> None:
    random.setstate(rng_state["python"])
    np_rng.bit_generator.state = rng_state["numpy"]
    torch.set_rng_state(rng_state["torch"])
    if rng_state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state["cuda"])
