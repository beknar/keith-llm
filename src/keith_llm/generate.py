"""KV-cache sampling: temperature, top-k, top-p, repetition penalty."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from keith_llm.model import Transformer


def sample_token(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_k: int | None = None,
    top_p: float = 0.95,
    generator: torch.Generator | None = None,
) -> int:
    """Sample one token id from a 1-D logits tensor."""
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k is not None and 0 < top_k < logits.shape[-1]:
        kth = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        cut = cumulative - sorted_probs >= top_p  # always keep the first token
        sorted_probs = sorted_probs.masked_fill(cut, 0.0)
        probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, num_samples=1, generator=generator))


def _apply_repetition_penalty(
    logits: torch.Tensor, seen_ids: Sequence[int], penalty: float
) -> torch.Tensor:
    if penalty == 1.0 or not seen_ids:
        return logits
    idx = torch.tensor(sorted(set(seen_ids)), device=logits.device)
    picked = logits[idx]
    logits[idx] = torch.where(picked > 0, picked / penalty, picked * penalty)
    return logits


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_ids: Sequence[int],
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_k: int | None = None,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    stop_ids: Sequence[int] = (),
    generator: torch.Generator | None = None,
) -> list[int]:
    """Generate token ids from a prompt using the KV cache (prefill once,
    then one token per step). Returns prompt + continuation; a generated
    stop id is included as the final element."""
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")
    if len(prompt_ids) >= model.cfg.max_seq_len:
        raise ValueError("prompt is already max_seq_len tokens")
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    stop = set(stop_ids)
    ids = list(prompt_ids)
    model.allocate_kv_cache(1, device=device, dtype=next(model.parameters()).dtype)
    try:
        logits, _ = model(torch.tensor([ids], device=device), start_pos=0)
        pos = len(ids)
        for _ in range(max_new_tokens):
            last = logits[0, -1].float()
            last = _apply_repetition_penalty(last, ids, repetition_penalty)
            next_id = sample_token(last, temperature, top_k, top_p, generator)
            ids.append(next_id)
            if next_id in stop or pos + 1 >= model.cfg.max_seq_len:
                break
            logits, _ = model(torch.tensor([[next_id]], device=device), start_pos=pos)
            pos += 1
    finally:
        model.free_kv_cache()
        if was_training:
            model.train()
    return ids
