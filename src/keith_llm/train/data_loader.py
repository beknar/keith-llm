"""Random-offset batch sampling from uint16 token bins.

The memmap is reopened on every call. This keeps steady-state RSS flat (the
page cache belongs to the kernel, not the process), which is what the soak
tests rely on when hunting real leaks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def get_batch(
    data_dir: str | Path,
    split: str,
    batch_size: int,
    seq_len: int,
    rng: np.random.Generator,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    path = Path(data_dir) / f"{split}.bin"
    data = np.memmap(path, dtype=np.uint16, mode="r")
    if len(data) < seq_len + 1:
        raise ValueError(f"{path} has {len(data)} tokens; need at least seq_len+1 = {seq_len + 1}")
    ix = rng.integers(0, len(data) - seq_len, size=batch_size)
    x = torch.stack([torch.from_numpy(data[i : i + seq_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + seq_len].astype(np.int64)) for i in ix])
    device = torch.device(device)
    if device.type == "cuda":
        return (
            x.pin_memory().to(device, non_blocking=True),
            y.pin_memory().to(device, non_blocking=True),
        )
    return x, y
