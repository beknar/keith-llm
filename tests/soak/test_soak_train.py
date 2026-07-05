"""Long training run; fails on sustained RSS or CUDA memory growth.

Run: pytest -m soak tests/soak/test_soak_train.py
Scale up (e.g. overnight on the GPU host): KEITH_SOAK_ITERS=20000
"""

import json
import os
import sys

import pytest
import torch

from keith_llm.config import ModelConfig, TrainConfig
from keith_llm.train.loop import Trainer

sys.path.insert(0, os.path.dirname(__file__))
from leakcheck import check_cuda_band, check_no_leak  # noqa: E402

pytestmark = pytest.mark.soak

SOAK_STEPS = int(os.environ.get("KEITH_SOAK_ITERS", "2000"))


def test_train_memory_stays_flat(tiny_bins, tmp_path):
    bins_dir, meta = tiny_bins
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg = ModelConfig(
        vocab_size=meta["vocab_size"],
        d_model=64,
        n_layers=2,
        n_heads=2,
        ffn_hidden=128,
        max_seq_len=128,
    )
    train_cfg = TrainConfig(
        batch_size=8,
        max_steps=SOAK_STEPS,
        warmup_steps=min(20, SOAK_STEPS),
        eval_interval=max(SOAK_STEPS // 10, 1),
        eval_batches=2,
        checkpoint_interval=max(SOAK_STEPS // 4, 1),
        dtype="bfloat16" if device == "cuda" else "float32",
        data_dir=str(bins_dir),
        out_dir=str(tmp_path / "soak_run"),
    )
    Trainer(model_cfg, train_cfg, device=device).train()

    metrics = [
        json.loads(ln) for ln in (tmp_path / "soak_run" / "metrics.jsonl").read_text().splitlines()
    ]
    steps = [m["step"] for m in metrics]
    rss = [m["rss_mb"] * 1e6 for m in metrics]
    check_no_leak(steps, rss, "trainer RSS")
    if device == "cuda":
        cuda = [m["cuda_alloc_mb"] * 1e6 for m in metrics]
        check_no_leak(steps, cuda, "trainer CUDA allocated", max_slope_per_iter=1e3)
        check_cuda_band(cuda, "trainer CUDA allocated")
