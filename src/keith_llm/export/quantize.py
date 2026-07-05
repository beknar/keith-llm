"""Quantization via llama.cpp's llama-quantize binary."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

QUANT_TYPES = ("Q8_0", "Q5_K_M", "Q4_K_M")


def quantize(
    gguf_in: str | Path,
    qtype: str,
    out_path: str | Path | None = None,
    bin_path: str | None = None,
) -> Path:
    """Run llama-quantize. K-quants need all tensor row sizes % 256 == 0
    (all GPU presets comply; tiny-1m is Q8_0-only)."""
    if qtype not in QUANT_TYPES:
        raise ValueError(f"qtype must be one of {QUANT_TYPES}, got {qtype!r}")
    gguf_in = Path(gguf_in)
    if out_path is None:
        stem = gguf_in.stem.removesuffix("-f16").removesuffix("-F16")
        out_path = gguf_in.with_name(f"{stem}-{qtype}.gguf")
    binary = bin_path or os.environ.get("LLAMA_QUANTIZE", "llama-quantize")
    if shutil.which(binary) is None and not Path(binary).exists():
        raise FileNotFoundError(
            f"llama-quantize not found ({binary!r}); build llama.cpp and/or set "
            "LLAMA_QUANTIZE to the binary path"
        )
    subprocess.run([str(binary), str(gguf_in), str(out_path), qtype], check=True)
    logger.info("quantized %s -> %s (%s)", gguf_in, out_path, qtype)
    return Path(out_path)
