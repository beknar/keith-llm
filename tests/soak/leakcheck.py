"""Leak-detection helpers for soak tests.

Methodology: sample memory over a long run, discard the warmup fraction
(allocator pools, cuDNN autotune, and torch.compile caches settle there),
fit a linear slope to the rest, and fail on sustained upward growth rather
than absolute usage. CUDA allocated memory additionally gets a max-min band
check, which catches stepwise leaks a regression line can smooth over.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field

import numpy as np
import psutil
import torch


@dataclass
class MemorySampler:
    use_cuda: bool = False
    iters: list[int] = field(default_factory=list)
    rss: list[float] = field(default_factory=list)
    cuda: list[float] = field(default_factory=list)

    def sample(self, iteration: int) -> None:
        gc.collect()
        self.iters.append(iteration)
        self.rss.append(float(psutil.Process().memory_info().rss))
        if self.use_cuda:
            torch.cuda.synchronize()
            self.cuda.append(float(torch.cuda.memory_allocated()))


def check_no_leak(
    iters: list[int],
    samples: list[float],
    label: str,
    warmup_frac: float = 0.2,
    max_slope_per_iter: float = 20e3,  # bytes/iter
    max_projected_growth: float = 50e6,  # bytes over the sampled window
    min_samples: int = 20,
) -> None:
    """Assert that post-warmup memory growth is bounded."""
    if len(samples) < min_samples:
        raise AssertionError(f"{label}: only {len(samples)} samples, need {min_samples}")
    cut = int(len(samples) * warmup_frac)
    xs = np.asarray(iters[cut:], dtype=float)
    ys = np.asarray(samples[cut:], dtype=float)
    slope = float(np.polyfit(xs, ys, 1)[0])
    projected = slope * (xs[-1] - xs[0])
    assert slope < max_slope_per_iter, (
        f"{label}: slope {slope / 1e3:.1f} KB/iter exceeds {max_slope_per_iter / 1e3:.1f} KB/iter"
    )
    assert projected < max_projected_growth, (
        f"{label}: projected growth {projected / 1e6:.1f} MB over window exceeds "
        f"{max_projected_growth / 1e6:.1f} MB"
    )


def check_cuda_band(
    samples: list[float], label: str, warmup_frac: float = 0.2, max_band: float = 1e6
) -> None:
    """Steady-state CUDA allocated memory must stay within a narrow band."""
    cut = int(len(samples) * warmup_frac)
    ys = np.asarray(samples[cut:], dtype=float)
    band = float(ys.max() - ys.min())
    assert band < max_band, (
        f"{label}: CUDA allocated varies by {band / 1e6:.2f} MB post-warmup "
        f"(limit {max_band / 1e6:.2f} MB)"
    )
