"""Repeated generation cycles; the fresh KV-cache allocation each cycle is
the likeliest GPU leak site.

Run: pytest -m soak tests/soak/test_soak_generate.py
Scale up: KEITH_SOAK_GEN_ITERS=5000
"""

import os
import sys

import pytest
import torch

from keith_llm.config import ModelConfig
from keith_llm.generate import generate
from keith_llm.model import Transformer

sys.path.insert(0, os.path.dirname(__file__))
from leakcheck import check_cuda_band, check_no_leak  # noqa: E402

pytestmark = pytest.mark.soak

CYCLES = int(os.environ.get("KEITH_SOAK_GEN_ITERS", "500"))
SAMPLE_EVERY = max(CYCLES // 100, 1)


def test_generate_memory_stays_flat():
    from leakcheck import MemorySampler

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ModelConfig(
        vocab_size=512, d_model=64, n_layers=2, n_heads=2, ffn_hidden=128, max_seq_len=256
    )
    torch.manual_seed(0)
    model = Transformer(cfg).to(device).eval()
    sampler = MemorySampler(use_cuda=device == "cuda")

    gen = torch.Generator(device=device).manual_seed(0)
    for cycle in range(CYCLES):
        generate(
            model,
            prompt_ids=[1, 2, 3],
            max_new_tokens=64,
            temperature=1.0,
            generator=gen,
        )
        if cycle % SAMPLE_EVERY == 0:
            sampler.sample(cycle)

    check_no_leak(sampler.iters, sampler.rss, "generate RSS")
    if device == "cuda":
        check_no_leak(sampler.iters, sampler.cuda, "generate CUDA", max_slope_per_iter=1e3)
        check_cuda_band(sampler.cuda, "generate CUDA")
