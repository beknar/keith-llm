"""Supervised fine-tuning: continue-train a pretrained base on instructions.

Loads a base checkpoint (its ``model_cfg`` defines the architecture), fine-tunes
on the loss-masked SFT examples for a few epochs at a low LR, and saves a
checkpoint in the same format as pretraining — so ``keith-llm export`` works on
it unchanged. Reuses the pretraining LR schedule, optimizer split, checkpoint
format, and JSONL metrics.
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from keith_llm.config import ModelConfig, TrainConfig
from keith_llm.model import Transformer
from keith_llm.sft.loader import load_sft_examples, sft_batches
from keith_llm.tokenizer.wrapper import KeithTokenizer
from keith_llm.train.checkpoint import load_checkpoint, save_checkpoint
from keith_llm.train.lr import cosine_warmup_lr
from keith_llm.train.metrics import JsonlLogger

logger = logging.getLogger(__name__)


class SFTTrainer:
    def __init__(
        self,
        base_ckpt: str | Path,
        data_jsonl: str | Path,
        tokenizer_path: str | Path,
        out_dir: str | Path,
        epochs: int = 3,
        lr: float = 2e-5,
        min_lr: float = 0.0,
        batch_size: int = 8,
        warmup_ratio: float = 0.03,
        weight_decay: float = 0.0,
        grad_clip: float = 1.0,
        seed: int = 1337,
        device: str | None = None,
        dtype: str = "bfloat16",
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self.batch_size = batch_size
        self.grad_clip = grad_clip

        torch.manual_seed(seed)
        random.seed(seed)
        self.np_rng = np.random.default_rng(seed)

        ckpt = load_checkpoint(base_ckpt, map_location=self.device)
        self.model_cfg = ModelConfig(**ckpt["model_cfg"])
        self.model = Transformer(self.model_cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        logger.info("loaded base %s (%d params)", base_ckpt, self.model.num_params())

        tok = KeithTokenizer.load(tokenizer_path)
        if tok.vocab_size != self.model_cfg.vocab_size:
            raise ValueError(
                f"tokenizer vocab ({tok.vocab_size}) != base model vocab "
                f"({self.model_cfg.vocab_size})"
            )
        self.pad_id = tok.pad_id
        self.examples = load_sft_examples(data_jsonl, tok, self.model_cfg.max_seq_len)
        if not self.examples:
            raise ValueError(f"no usable SFT examples in {data_jsonl}")
        steps_per_epoch = math.ceil(len(self.examples) / batch_size)
        self.total_steps = max(1, epochs * steps_per_epoch)

        # Reuse the pretraining cosine schedule via a TrainConfig shell.
        self.lr_cfg = TrainConfig(
            lr=lr,
            min_lr=min_lr,
            warmup_steps=max(1, int(self.total_steps * warmup_ratio)),
            max_steps=self.total_steps,
        )
        self.optimizer = self._make_optimizer(weight_decay, lr)
        self.autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if dtype == "bfloat16" and self.device.type == "cuda"
            else torch.autocast(device_type="cpu", enabled=False)
        )
        self.logger = JsonlLogger(self.out_dir / "metrics.jsonl")

    def _make_optimizer(self, weight_decay: float, lr: float) -> torch.optim.AdamW:
        decay = [p for p in self.model.parameters() if p.dim() >= 2]
        no_decay = [p for p in self.model.parameters() if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), fused=self.device.type == "cuda")

    def train(self) -> float:
        self.model.train()
        step = 0
        last_loss = float("nan")
        for epoch in range(self.epochs):
            for x, y in sft_batches(
                self.examples, self.batch_size, self.pad_id, self.np_rng, self.device
            ):
                lr = cosine_warmup_lr(step, self.lr_cfg)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                t0 = time.perf_counter()
                with self.autocast:
                    _, loss = self.model(x, targets=y)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                step += 1
                last_loss = loss.item()
                self.logger.log(
                    step=step,
                    epoch=epoch + 1,
                    loss=last_loss,
                    lr=lr,
                    sec_per_step=time.perf_counter() - t0,
                )
        save_checkpoint(
            self.out_dir / "latest.pt",
            self.model,
            self.optimizer,
            step,
            self.model_cfg,
            replace(self.lr_cfg, out_dir=str(self.out_dir)),
            self.np_rng,
        )
        self.logger.close()
        logger.info("SFT done: %d steps, final loss %.4f -> %s", step, last_loss, self.out_dir)
        return last_loss
