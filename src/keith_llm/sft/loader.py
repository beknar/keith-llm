"""Tokenize SFT examples with loss masking, and batch them.

Each example becomes ``<|bos|>`` + prompt + response + ``<|eos|>``. The target
is the sequence shifted by one, with **prompt positions masked to -1** so loss
is computed only over the response and the final ``<|eos|>`` — the model learns
to *produce* the answer and to stop, not to echo the instruction. The model's
cross-entropy already uses ``ignore_index=-1``.

Batches are right-padded to the batch's longest example (pad ids in the input,
-1 in the target). With causal attention and right padding, real tokens never
attend to pad and pad predictions carry no loss, so no attention mask is needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from keith_llm.sft.format import build_prompt
from keith_llm.tokenizer.wrapper import KeithTokenizer

Example = tuple[list[int], list[int]]  # (input_ids, target_ids)


def tokenize_example(
    tok: KeithTokenizer, instruction: str, response: str, max_seq_len: int
) -> Example | None:
    """Return (input_ids, target_ids) or None if no response survives truncation."""
    prompt_ids = [tok.bos_id] + tok.encode(build_prompt(instruction))
    response_ids = tok.encode(response) + [tok.eos_id]
    full = (prompt_ids + response_ids)[:max_seq_len]
    if len(full) < 2:
        return None
    n_prompt = len(prompt_ids)
    x = full[:-1]
    y = full[1:]
    # y[i] predicts full[i+1]; that's a response token when i+1 >= n_prompt.
    # Mask everything before that (predictions of prompt tokens).
    for i in range(min(n_prompt - 1, len(y))):
        y[i] = -1
    if all(t == -1 for t in y):  # response truncated away entirely
        return None
    return x, y


def load_sft_examples(
    jsonl_path: str | Path, tok: KeithTokenizer, max_seq_len: int
) -> list[Example]:
    examples: list[Example] = []
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ex = tokenize_example(tok, rec["instruction"], rec["response"], max_seq_len)
            if ex is not None:
                examples.append(ex)
    return examples


def sft_batches(
    examples: list[Example],
    batch_size: int,
    pad_id: int,
    rng: np.random.Generator,
    device: str | torch.device = "cpu",
    shuffle: bool = True,
):
    """Yield right-padded (x, y) int64 tensors. One pass over the examples."""
    order = rng.permutation(len(examples)) if shuffle else np.arange(len(examples))
    device = torch.device(device)
    for start in range(0, len(examples), batch_size):
        batch = [examples[i] for i in order[start : start + batch_size]]
        maxlen = max(len(x) for x, _ in batch)
        xs = torch.full((len(batch), maxlen), pad_id, dtype=torch.int64)
        ys = torch.full((len(batch), maxlen), -1, dtype=torch.int64)
        for row, (x, y) in enumerate(batch):
            xs[row, : len(x)] = torch.tensor(x, dtype=torch.int64)
            ys[row, : len(y)] = torch.tensor(y, dtype=torch.int64)
        if device.type == "cuda":
            yield (
                xs.pin_memory().to(device, non_blocking=True),
                ys.pin_memory().to(device, non_blocking=True),
            )
        else:
            yield xs.to(device), ys.to(device)
