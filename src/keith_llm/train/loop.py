"""The Trainer: bf16 autocast, grad accumulation, eval, sampling, checkpoints."""

from __future__ import annotations

import contextlib
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from keith_llm.config import ModelConfig, TrainConfig
from keith_llm.model import Transformer
from keith_llm.train.checkpoint import load_checkpoint, restore_rng, save_checkpoint
from keith_llm.train.data_loader import get_batch
from keith_llm.train.lr import cosine_warmup_lr
from keith_llm.train.metrics import JsonlLogger

logger = logging.getLogger(__name__)

_EVAL_SEED = 1234  # fixed so every eval sees the same val batches


class Trainer:
    def __init__(
        self,
        model_cfg: ModelConfig,
        train_cfg: TrainConfig,
        device: str | None = None,
        resume: str | Path | None = None,
        tokenizer_path: str | Path | None = None,
    ):
        self.model_cfg = model_cfg
        self.cfg = train_cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.out_dir = Path(train_cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer_path = tokenizer_path

        meta_path = Path(train_cfg.data_dir) / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta["vocab_size"] > model_cfg.vocab_size:
                raise ValueError(
                    f"token bins use vocab {meta['vocab_size']} > model vocab "
                    f"{model_cfg.vocab_size}"
                )

        torch.manual_seed(train_cfg.seed)
        random.seed(train_cfg.seed)
        self.np_rng = np.random.default_rng(train_cfg.seed)

        self.model = Transformer(model_cfg).to(self.device)
        self.optimizer = self._make_optimizer()
        self.step = 0
        if resume is not None:
            ckpt = load_checkpoint(resume, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.step = ckpt["step"]
            restore_rng(ckpt["rng"], self.np_rng)
            logger.info("resumed from %s at step %d", resume, self.step)

        self.fwd = self.model
        if train_cfg.compile and self.device.type == "cuda":
            self.fwd = torch.compile(self.model)

        self.autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if train_cfg.dtype == "bfloat16" and self.device.type == "cuda"
            else contextlib.nullcontext()
        )
        self.logger = JsonlLogger(self.out_dir / "metrics.jsonl")
        self.process = psutil.Process()

    def _make_optimizer(self) -> torch.optim.AdamW:
        decay, no_decay = [], []
        for param in self.model.parameters():
            (decay if param.dim() >= 2 else no_decay).append(param)
        groups = [
            {"params": decay, "weight_decay": self.cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(
            groups,
            lr=self.cfg.lr,
            betas=(0.9, 0.95),
            fused=self.device.type == "cuda",
        )

    def _batch(self, split: str, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        return get_batch(
            self.cfg.data_dir,
            split,
            self.cfg.batch_size,
            self.model_cfg.max_seq_len,
            rng,
            device=self.device,
        )

    @torch.no_grad()
    def evaluate(self) -> float:
        rng = np.random.default_rng(_EVAL_SEED)
        self.model.eval()
        losses = []
        for _ in range(self.cfg.eval_batches):
            x, y = self._batch("val", rng)
            with self.autocast:
                _, loss = self.fwd(x, y)
            losses.append(loss.item())
        self.model.train()
        return float(np.mean(losses))

    def _sample(self, step: int) -> None:
        if self.tokenizer_path is None or not Path(self.tokenizer_path).exists():
            return
        from keith_llm.constants import SYSTEMS
        from keith_llm.generate import generate
        from keith_llm.tokenizer.wrapper import KeithTokenizer

        tok = KeithTokenizer.load(self.tokenizer_path)
        lines = [f"=== step {step} ==="]
        for system in SYSTEMS:
            prompt = tok.control_prefix(system, "adventure")
            ids = generate(self.model, prompt, max_new_tokens=120, stop_ids=[tok.eos_id])
            lines.append(f"--- {system} ---\n{tok.decode(ids)}\n")
        with (self.out_dir / "samples.txt").open("a") as fh:
            fh.write("\n".join(lines) + "\n")

    def _checkpoint(self, step: int) -> None:
        save_checkpoint(
            self.out_dir / "latest.pt",
            self.model,
            self.optimizer,
            step,
            self.model_cfg,
            self.cfg,
            self.np_rng,
        )

    def train(self) -> float:
        """Run to max_steps; returns the final training loss."""
        cfg = self.cfg
        tokens_per_step = cfg.batch_size * cfg.grad_accum * self.model_cfg.max_seq_len
        self.model.train()
        last_loss = float("nan")
        for step in range(self.step, cfg.max_steps):
            lr = cosine_warmup_lr(step, cfg)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            t0 = time.perf_counter()
            loss_accum = 0.0
            for _ in range(cfg.grad_accum):
                x, y = self._batch("train", self.np_rng)
                with self.autocast:
                    _, loss = self.fwd(x, y)
                loss = loss / cfg.grad_accum
                loss_accum += loss.item()
                loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            dt = time.perf_counter() - t0
            last_loss = loss_accum

            done = step + 1
            val_loss = None
            if cfg.eval_interval and done % cfg.eval_interval == 0:
                val_loss = self.evaluate()
            self.logger.log(
                step=done,
                loss=loss_accum,
                val_loss=val_loss,
                lr=lr,
                tok_per_sec=tokens_per_step / dt,
                rss_mb=self.process.memory_info().rss / 1e6,
                cuda_alloc_mb=(
                    torch.cuda.memory_allocated() / 1e6 if self.device.type == "cuda" else None
                ),
            )
            if cfg.sample_interval and done % cfg.sample_interval == 0:
                self._sample(done)
            if cfg.checkpoint_interval and done % cfg.checkpoint_interval == 0:
                self._checkpoint(done)
            self.step = done

        self._checkpoint(self.step)
        self.logger.close()
        return last_loss
