"""Tokenize the corpus into flat uint16 train/val bins (np.memmap-ready).

Each document is written as ``[bos, <|system:X|>, <|doc:Y|>] + tokens + [eos]``
into one contiguous stream. The val split is per-document and deterministic
(stable across runs and machines); if no document lands in val, the smallest
document id is promoted so val.bin is never empty for multi-doc corpora.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from keith_llm.tokenizer.wrapper import KeithTokenizer

logger = logging.getLogger(__name__)


def _is_val(doc_id: str, val_mod: int) -> bool:
    return int(doc_id[:8], 16) % val_mod == 0


def binarize(
    corpus_jsonl: str | Path,
    tokenizer_path: str | Path,
    out_dir: str | Path,
    val_mod: int = 50,
) -> dict[str, Any]:
    tok = KeithTokenizer.load(tokenizer_path)
    if tok.vocab_size >= 2**16:
        raise ValueError("vocab too large for uint16 bins")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: decide the val set from doc ids and sizes.
    doc_sizes: dict[str, int] = {}
    with open(corpus_jsonl) as fh:
        for line in fh:
            rec = json.loads(line)
            doc_sizes[rec["id"]] = len(rec["text"])
    if not doc_sizes:
        raise ValueError(f"empty corpus: {corpus_jsonl}")
    val_ids = {d for d in doc_sizes if _is_val(d, val_mod)}
    if not val_ids and len(doc_sizes) > 1:
        # Promote the SMALLEST document: on few-document corpora this keeps
        # the training set nearly intact while still giving val a real doc.
        smallest = min(doc_sizes, key=lambda d: (doc_sizes[d], d))
        val_ids = {smallest}
        logger.warning(
            "no document hashed into val; promoted smallest doc %s (%d chars)",
            smallest[:12],
            doc_sizes[smallest],
        )

    # Pass 2: encode and stream to the bins.
    counts = {"train": 0, "val": 0}
    with (
        open(corpus_jsonl) as fh,
        open(out_dir / "train.bin", "wb") as f_train,
        open(out_dir / "val.bin", "wb") as f_val,
    ):
        for line in fh:
            rec = json.loads(line)
            ids = (
                tok.control_prefix(rec["system"], rec["doc_type"])
                + tok.encode(rec["text"])
                + [tok.eos_id]
            )
            split = "val" if rec["id"] in val_ids else "train"
            counts[split] += len(ids)
            (f_val if split == "val" else f_train).write(np.asarray(ids, dtype=np.uint16).tobytes())

    meta = {
        "dtype": "uint16",
        "vocab_size": tok.vocab_size,
        "n_train_tokens": counts["train"],
        "n_val_tokens": counts["val"],
        "n_documents": len(doc_sizes),
        "n_val_documents": len(val_ids),
        "tokenizer_sha1": hashlib.sha1(Path(tokenizer_path).read_bytes()).hexdigest(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    logger.info("binarized -> %s: %s", out_dir, meta)
    return meta
