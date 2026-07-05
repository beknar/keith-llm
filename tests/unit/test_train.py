import json

import numpy as np
import pytest
import torch

from keith_llm.config import ModelConfig, TrainConfig
from keith_llm.model import Transformer
from keith_llm.train.data_loader import get_batch
from keith_llm.train.loop import Trainer
from keith_llm.train.lr import cosine_warmup_lr


def _train_cfg(bins_dir, out_dir, **overrides):
    defaults = dict(
        batch_size=4,
        grad_accum=1,
        max_steps=6,
        lr=1e-3,
        min_lr=1e-4,
        warmup_steps=2,
        weight_decay=0.1,
        grad_clip=1.0,
        eval_interval=0,
        eval_batches=2,
        sample_interval=0,
        checkpoint_interval=3,
        dtype="float32",
        compile=False,
        seed=7,
        data_dir=str(bins_dir),
        out_dir=str(out_dir),
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _model_cfg(vocab_size):
    return ModelConfig(
        vocab_size=vocab_size, d_model=32, n_layers=2, n_heads=2, ffn_hidden=64, max_seq_len=64
    )


# --- LR schedule ---


def test_lr_warmup_start():
    cfg = TrainConfig(lr=1e-3, min_lr=1e-4, warmup_steps=10, max_steps=100)
    assert cosine_warmup_lr(0, cfg) == pytest.approx(1e-4)  # (0+1)/10 * lr


def test_lr_full_after_warmup():
    cfg = TrainConfig(lr=1e-3, min_lr=1e-4, warmup_steps=10, max_steps=100)
    assert cosine_warmup_lr(10, cfg) == pytest.approx(1e-3)


def test_lr_min_at_end():
    cfg = TrainConfig(lr=1e-3, min_lr=1e-4, warmup_steps=10, max_steps=100)
    assert cosine_warmup_lr(100, cfg) == pytest.approx(1e-4)
    assert cosine_warmup_lr(500, cfg) == pytest.approx(1e-4)


def test_lr_midpoint():
    cfg = TrainConfig(lr=1e-3, min_lr=0.0, warmup_steps=0, max_steps=100)
    assert cosine_warmup_lr(50, cfg) == pytest.approx(5e-4)


# --- data loader ---


def test_get_batch_shapes_and_shift(tiny_bins):
    bins_dir, _ = tiny_bins
    rng = np.random.default_rng(0)
    x, y = get_batch(bins_dir, "train", batch_size=3, seq_len=16, rng=rng)
    assert x.shape == y.shape == (3, 16)
    assert x.dtype == torch.int64
    data = np.memmap(bins_dir / "train.bin", dtype=np.uint16, mode="r")
    # y is x shifted by one within the flat stream
    row = x[0].numpy().astype(np.uint16)
    starts = np.flatnonzero(
        np.all(np.lib.stride_tricks.sliding_window_view(np.asarray(data), 16) == row, axis=1)
    )
    assert any(np.array_equal(data[s + 1 : s + 17], y[0].numpy().astype(np.uint16)) for s in starts)


def test_get_batch_rejects_short_bin(tmp_path):
    (tmp_path / "train.bin").write_bytes(np.arange(4, dtype=np.uint16).tobytes())
    with pytest.raises(ValueError, match="seq_len"):
        get_batch(tmp_path, "train", batch_size=1, seq_len=16, rng=np.random.default_rng(0))


# --- grad accumulation math ---


def test_grad_accum_equivalence():
    cfg = _model_cfg(64)
    torch.manual_seed(0)
    model_a = Transformer(cfg)
    model_b = Transformer(cfg)
    model_b.load_state_dict(model_a.state_dict())
    x = torch.randint(0, 64, (4, 16))
    y = torch.randint(0, 64, (4, 16))

    _, loss = model_a(x, y)
    loss.backward()

    for half in (slice(0, 2), slice(2, 4)):
        _, loss = model_b(x[half], y[half])
        (loss / 2).backward()

    for pa, pb in zip(model_a.parameters(), model_b.parameters(), strict=True):
        assert torch.allclose(pa.grad, pb.grad, atol=1e-5)


# --- Trainer ---


def test_training_reduces_loss_and_logs(tiny_bins, tmp_path):
    bins_dir, meta = tiny_bins
    cfg = _train_cfg(bins_dir, tmp_path / "run", max_steps=30, eval_interval=15, lr=3e-3)
    trainer = Trainer(_model_cfg(meta["vocab_size"]), cfg, device="cpu")
    trainer.train()

    lines = [json.loads(ln) for ln in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert len(lines) == 30
    assert lines[0]["loss"] > lines[-1]["loss"], "loss should drop over 30 steps"
    assert {"step", "loss", "lr", "tok_per_sec", "rss_mb"} <= lines[0].keys()
    assert "val_loss" in lines[14]  # eval at step 15
    assert (tmp_path / "run" / "latest.pt").exists()


def test_checkpoint_resume_bit_exact(tiny_bins, tmp_path):
    bins_dir, meta = tiny_bins
    mcfg = _model_cfg(meta["vocab_size"])

    # Uninterrupted 6-step run (checkpoint written at step 3).
    cfg_a = _train_cfg(bins_dir, tmp_path / "a")
    trainer_a = Trainer(mcfg, cfg_a, device="cpu")
    loss_a = trainer_a.train()

    # Fresh 3-step run, then resume from its checkpoint for the last 3.
    cfg_b3 = _train_cfg(bins_dir, tmp_path / "b", max_steps=3)
    Trainer(mcfg, cfg_b3, device="cpu").train()
    cfg_b6 = _train_cfg(bins_dir, tmp_path / "b", max_steps=6)
    trainer_b = Trainer(mcfg, cfg_b6, device="cpu", resume=tmp_path / "b" / "latest.pt")
    assert trainer_b.step == 3
    loss_b = trainer_b.train()

    assert loss_a == loss_b
    state_a = trainer_a.model.state_dict()
    state_b = trainer_b.model.state_dict()
    for key in state_a:
        assert torch.equal(state_a[key], state_b[key]), key


def test_eval_is_deterministic(tiny_bins, tmp_path):
    bins_dir, meta = tiny_bins
    cfg = _train_cfg(bins_dir, tmp_path / "run", max_steps=1, warmup_steps=1)
    trainer = Trainer(_model_cfg(meta["vocab_size"]), cfg, device="cpu")
    assert trainer.evaluate() == trainer.evaluate()


def test_trainer_rejects_vocab_mismatch(tiny_bins, tmp_path):
    bins_dir, meta = tiny_bins
    cfg = _train_cfg(bins_dir, tmp_path / "run")
    with pytest.raises(ValueError, match="vocab"):
        Trainer(_model_cfg(meta["vocab_size"] - 100), cfg, device="cpu")


def test_overfit_tiny_text(tiny_bins, tmp_path):
    """~150 steps on the tiny corpus should drive train loss well below the
    uniform baseline (ln(vocab) ~ 6)."""
    bins_dir, meta = tiny_bins
    cfg = _train_cfg(
        bins_dir, tmp_path / "run", max_steps=150, lr=3e-3, warmup_steps=10, batch_size=8
    )
    trainer = Trainer(_model_cfg(meta["vocab_size"]), cfg, device="cpu")
    final_loss = trainer.train()
    assert final_loss < 1.5, f"expected memorization, got loss {final_loss}"
