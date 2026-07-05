"""Checkpoint -> f16 GGUF with the ``llama`` architecture.

Works out-of-the-box in llama.cpp/ollama because:
- the model's RoPE is the Meta interleaved convention (ROPE_TYPE_NORM), so
  q/k weights are written as-is, no permutation;
- the tokenizer is byte-level BPE end-to-end, so tokenizer.json's vocab and
  merges map directly onto llama.cpp's ``gpt2`` tokenizer.

``tokenizer.ggml.pre`` must be set (``gpt-2``) or llama.cpp falls back to a
"default" pre-tokenizer with subtly different splits.

BOS handling: ``add_bos_token=true`` means llama.cpp prepends <|bos|> itself.
Prompts fed to llama.cpp/ollama should therefore start with the control
tokens (``<|system:X|><|doc:Y|>``) but NOT include <|bos|>.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import gguf
import numpy as np
import torch

from keith_llm.config import ModelConfig
from keith_llm.tokenizer.wrapper import BOS, EOS, PAD

logger = logging.getLogger(__name__)

# state_dict key template -> GGUF tensor name (per-block)
_BLOCK_TENSORS = {
    "blocks.{i}.attention_norm.weight": "blk.{i}.attn_norm.weight",
    "blocks.{i}.attention.wq.weight": "blk.{i}.attn_q.weight",
    "blocks.{i}.attention.wk.weight": "blk.{i}.attn_k.weight",
    "blocks.{i}.attention.wv.weight": "blk.{i}.attn_v.weight",
    "blocks.{i}.attention.wo.weight": "blk.{i}.attn_output.weight",
    "blocks.{i}.ffn_norm.weight": "blk.{i}.ffn_norm.weight",
    "blocks.{i}.feed_forward.w_gate.weight": "blk.{i}.ffn_gate.weight",
    "blocks.{i}.feed_forward.w_up.weight": "blk.{i}.ffn_up.weight",
    "blocks.{i}.feed_forward.w_down.weight": "blk.{i}.ffn_down.weight",
}


def _load_vocab(tokenizer_path: str | Path) -> tuple[list[str], list[int], list[str]]:
    """Return (tokens in id order, token types, merges) from tokenizer.json."""
    tj = json.loads(Path(tokenizer_path).read_text())
    vocab: dict[str, int] = tj["model"]["vocab"]
    added = {t["content"]: t["id"] for t in tj.get("added_tokens", [])}
    size = max(list(vocab.values()) + list(added.values())) + 1

    tokens = [""] * size
    types = [int(gguf.TokenType.NORMAL)] * size
    filled = [False] * size
    for tok, tid in vocab.items():
        tokens[tid] = tok
        filled[tid] = True
    for tok, tid in added.items():
        tokens[tid] = tok
        types[tid] = int(gguf.TokenType.CONTROL)
        filled[tid] = True
    holes = [i for i, ok in enumerate(filled) if not ok]
    if holes:
        raise ValueError(f"vocabulary has holes at ids {holes[:5]}")

    merges_raw = tj["model"]["merges"]
    merges = [m if isinstance(m, str) else " ".join(m) for m in merges_raw]
    return tokens, types, merges


def export_gguf(
    ckpt_path: str | Path,
    tokenizer_path: str | Path,
    out_path: str | Path,
    name: str = "keith-llm",
) -> Path:
    from keith_llm.train.checkpoint import load_checkpoint

    ckpt = load_checkpoint(ckpt_path)
    cfg = ModelConfig(**ckpt["model_cfg"])
    state: dict[str, torch.Tensor] = ckpt["model_state"]

    tokens, types, merges = _load_vocab(tokenizer_path)
    if len(tokens) > cfg.vocab_size:
        raise ValueError(f"tokenizer vocab {len(tokens)} exceeds model vocab {cfg.vocab_size}")
    # Model embeddings may be padded past the tokenizer vocab; fill with
    # explicit unused slots so token count matches the tensor dimension.
    while len(tokens) < cfg.vocab_size:
        types.append(int(gguf.TokenType.UNUSED))
        tokens.append(f"<|unused_{len(tokens)}|>")

    bos_id = tokens.index(BOS)
    eos_id = tokens.index(EOS)
    pad_id = tokens.index(PAD)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(out_path, arch="llama")
    writer.add_name(name)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    writer.add_context_length(cfg.max_seq_len)
    writer.add_embedding_length(cfg.d_model)
    writer.add_block_count(cfg.n_layers)
    writer.add_feed_forward_length(cfg.ffn_hidden)
    writer.add_head_count(cfg.n_heads)
    writer.add_head_count_kv(cfg.n_heads)
    writer.add_layer_norm_rms_eps(cfg.norm_eps)
    writer.add_rope_freq_base(cfg.rope_theta)
    writer.add_rope_dimension_count(cfg.head_dim)
    writer.add_vocab_size(cfg.vocab_size)

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("gpt-2")
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    writer.add_token_merges(merges)
    writer.add_bos_token_id(bos_id)
    writer.add_eos_token_id(eos_id)
    writer.add_pad_token_id(pad_id)
    writer.add_add_bos_token(True)

    def put(state_key: str, gguf_name: str) -> None:
        tensor = state[state_key].float().cpu().numpy()
        if tensor.ndim >= 2:
            tensor = tensor.astype(np.float16)
        writer.add_tensor(gguf_name, tensor)

    put("tok_emb.weight", "token_embd.weight")
    for i in range(cfg.n_layers):
        for key_tpl, name_tpl in _BLOCK_TENSORS.items():
            put(key_tpl.format(i=i), name_tpl.format(i=i))
    put("norm.weight", "output_norm.weight")
    # Explicit output head (duplicate of the tied embedding) — don't rely on
    # llama.cpp's tie fallback.
    put("lm_head.weight", "output.weight")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    logger.info("wrote %s (%d tokens, %d layers)", out_path, cfg.vocab_size, cfg.n_layers)
    return out_path
