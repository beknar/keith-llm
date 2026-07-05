"""Configuration dataclasses and YAML loading.

A config file has two top-level sections, ``model:`` and ``train:``, whose keys
map 1:1 onto :class:`ModelConfig` and :class:`TrainConfig` fields. Unknown keys
are rejected so typos fail loudly instead of silently using defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 16384
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 2
    ffn_hidden: int = 256
    max_seq_len: int = 256
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.vocab_size >= 2**16:
            raise ValueError("vocab_size must fit in uint16 token bins (< 65536)")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


@dataclass
class TrainConfig:
    batch_size: int = 8
    grad_accum: int = 1
    max_steps: int = 200
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 20
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 10
    sample_interval: int = 0  # 0 disables periodic sample generation
    checkpoint_interval: int = 100
    dtype: str = "float32"  # float32 | bfloat16
    compile: bool = False
    seed: int = 1337
    data_dir: str = "data/tokens"
    out_dir: str = "checkpoints/run"

    def __post_init__(self) -> None:
        if self.dtype not in ("float32", "bfloat16"):
            raise ValueError(f"dtype must be float32 or bfloat16, got {self.dtype!r}")
        if self.warmup_steps > self.max_steps:
            raise ValueError("warmup_steps must not exceed max_steps")


def _build(cls: type, section: dict[str, Any], name: str) -> Any:
    valid = {f.name for f in fields(cls)}
    unknown = set(section) - valid
    if unknown:
        raise ValueError(f"unknown keys in '{name}' section: {sorted(unknown)}")
    return cls(**section)


def load_config(path: str | Path) -> tuple[ModelConfig, TrainConfig]:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown = set(raw) - {"model", "train"}
    if unknown:
        raise ValueError(f"unknown top-level sections: {sorted(unknown)}")
    model = _build(ModelConfig, raw.get("model", {}), "model")
    train = _build(TrainConfig, raw.get("train", {}), "train")
    return model, train
