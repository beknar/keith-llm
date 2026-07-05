"""Llama-style decoder-only transformer in a single readable file.

Architecture: RMSNorm (pre-norm), rotary position embeddings, SwiGLU MLP,
multi-head attention with optional preallocated KV cache, tied embeddings.

RoPE uses the Meta/llama2.c interleaved-pair convention (rotate adjacent
element pairs via complex multiplication). llama.cpp's ``llama`` architecture
applies exactly this rotation (ROPE_TYPE_NORM), so GGUF export requires no
q/k weight permutation. Do not switch to the HF rotate-half convention.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from keith_llm.config import ModelConfig


def precompute_freqs_cis(head_dim: int, max_seq_len: int, theta: float) -> torch.Tensor:
    """Complex rotation factors, shape (max_seq_len, head_dim // 2)."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Rotate interleaved pairs. x: (B, T, n_heads, head_dim);
    freqs_cis: (T, head_dim // 2)."""
    x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs_cis.view(1, x_c.shape[1], 1, x_c.shape[-1])
    out = torch.view_as_real(x_c * freqs).flatten(3)
    return out.type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        x_f = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x_f.type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # Set by Transformer.allocate_kv_cache(); shape (B, max_seq, H, D).
        self.cache_k: torch.Tensor | None = None
        self.cache_v: torch.Tensor | None = None

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int | None
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        q = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        k = self.wk(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        v = self.wv(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        q = apply_rotary(q, freqs_cis)
        k = apply_rotary(k, freqs_cis)

        if start_pos is None:
            # Training / plain forward: causal attention over the sequence.
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            # Incremental decode against the preallocated cache. Supported
            # shapes: full prefill (start_pos == 0) or one token at a time.
            if self.cache_k is None or self.cache_v is None:
                raise RuntimeError("KV cache not allocated; call allocate_kv_cache() first")
            if seqlen > 1 and start_pos != 0:
                raise ValueError("multi-token forward requires start_pos == 0")
            self.cache_k[:bsz, start_pos : start_pos + seqlen] = k
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = v
            k_all = self.cache_k[:bsz, : start_pos + seqlen]
            v_all = self.cache_v[:bsz, : start_pos + seqlen]
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k_all.transpose(1, 2),
                v_all.transpose(1, 2),
                is_causal=seqlen > 1,
            )
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.resid_dropout(self.wo(out))


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w_down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attention = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.feed_forward = FeedForward(cfg)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int | None
    ) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), freqs_cis, start_pos)
        return x + self.feed_forward(self.ffn_norm(x))


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.emb_dropout = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta),
            persistent=False,
        )
        self.apply(self._init_weights)
        # GPT-2-style scaled init on residual-stream projections.
        resid_std = 0.02 / math.sqrt(2 * cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attention.wo.weight, mean=0.0, std=resid_std)
            nn.init.normal_(block.feed_forward.w_down.weight, mean=0.0, std=resid_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def allocate_kv_cache(
        self, batch_size: int, device: torch.device | str, dtype: torch.dtype = torch.float32
    ) -> None:
        shape = (batch_size, self.cfg.max_seq_len, self.cfg.n_heads, self.cfg.head_dim)
        for block in self.blocks:
            block.attention.cache_k = torch.zeros(shape, device=device, dtype=dtype)
            block.attention.cache_v = torch.zeros(shape, device=device, dtype=dtype)

    def free_kv_cache(self) -> None:
        for block in self.blocks:
            block.attention.cache_k = None
            block.attention.cache_v = None

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        start_pos: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        seqlen = idx.shape[1]
        offset = start_pos or 0
        if offset + seqlen > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence of {offset + seqlen} exceeds max_seq_len {self.cfg.max_seq_len}"
            )
        x = self.emb_dropout(self.tok_emb(idx))
        freqs_cis = self.freqs_cis[offset : offset + seqlen]
        for block in self.blocks:
            x = block(x, freqs_cis, start_pos)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-1
            )
        return logits, loss
