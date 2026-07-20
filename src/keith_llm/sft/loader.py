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
Turn = tuple[str, str]  # (user, assistant)


def tokenize_conversation(
    tok: KeithTokenizer, turns: list[Turn], max_seq_len: int
) -> Example | None:
    """Tokenize a one- or multi-turn conversation with loss masking.

    Builds ``[bos]`` + for each turn ``prompt(q) + a + [eos]``, concatenated.
    Loss is computed over **every** assistant response and its ``[eos]`` (so the
    model learns to answer each turn given the whole preceding conversation and
    to stop), and masked over ``[bos]`` and all instruction/prompt tokens.
    Returns None if nothing loss-bearing survives truncation."""
    full: list[int] = [tok.bos_id]
    is_target: list[bool] = [False]  # True where the token is an assistant/eos token
    for user, assistant in turns:
        p_ids = tok.encode(build_prompt(user))
        full += p_ids
        is_target += [False] * len(p_ids)
        a_ids = tok.encode(assistant.strip()) + [tok.eos_id]
        full += a_ids
        is_target += [True] * len(a_ids)
    full = full[:max_seq_len]
    is_target = is_target[:max_seq_len]
    if len(full) < 2:
        return None
    x = full[:-1]
    # y[i] predicts full[i+1]; keep it only when full[i+1] is an assistant token.
    y = [full[i + 1] if is_target[i + 1] else -1 for i in range(len(full) - 1)]
    if all(t == -1 for t in y):  # all responses truncated away
        return None
    return x, y


def tokenize_example(
    tok: KeithTokenizer, instruction: str, response: str, max_seq_len: int
) -> Example | None:
    """Single-turn convenience wrapper around :func:`tokenize_conversation`."""
    return tokenize_conversation(tok, [(instruction, response)], max_seq_len)


def _record_to_turns(rec: dict) -> list[Turn]:
    """Turn an SFT record into (user, assistant) pairs. Accepts single-turn
    ``{"instruction","response"}`` or multi-turn ``{"messages":[{role,content}]}``."""
    if "messages" in rec:
        turns: list[Turn] = []
        pending_user: str | None = None
        for msg in rec["messages"]:
            role, content = msg.get("role"), str(msg.get("content", ""))
            if role == "user":
                pending_user = content
            elif role == "assistant" and pending_user is not None:
                turns.append((pending_user, content))
                pending_user = None
        return turns
    return [(rec["instruction"], rec["response"])]


def load_sft_examples(
    jsonl_path: str | Path, tok: KeithTokenizer, max_seq_len: int
) -> list[Example]:
    examples: list[Example] = []
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            turns = _record_to_turns(json.loads(line))
            if not turns:
                continue
            ex = tokenize_conversation(tok, turns, max_seq_len)
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
