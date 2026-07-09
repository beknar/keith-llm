import json

import numpy as np
import torch

from keith_llm.config import ModelConfig, TrainConfig
from keith_llm.model import Transformer
from keith_llm.sft.loader import load_sft_examples, sft_batches, tokenize_example
from keith_llm.sft.trainer import SFTTrainer
from keith_llm.tokenizer.wrapper import KeithTokenizer
from keith_llm.train.checkpoint import load_checkpoint, save_checkpoint

# --- loss masking ---


def test_tokenize_masks_prompt_only(tiny_tokenizer_path):
    tok = KeithTokenizer.load(tiny_tokenizer_path)
    x, y = tokenize_example(tok, "What is the AC of a goblin?", "It is 15.", max_seq_len=128)
    assert len(x) == len(y)
    # Some leading targets masked (prompt), some real (response), ending in eos.
    assert y[0] == -1
    real = [t for t in y if t != -1]
    assert real, "response tokens must contribute loss"
    assert real[-1] == tok.eos_id  # learns to stop
    # the boundary: exactly the prompt predictions are masked
    n_masked = sum(1 for t in y if t == -1)
    assert 0 < n_masked < len(y)


def test_tokenize_truncates_and_drops_responseless(tiny_tokenizer_path):
    tok = KeithTokenizer.load(tiny_tokenizer_path)
    # tiny max_seq_len so the prompt fills it and the response is cut away
    out = tokenize_example(tok, "word " * 50, "the answer", max_seq_len=8)
    assert out is None


def test_load_sft_examples(tiny_tokenizer_path, tmp_path):
    tok = KeithTokenizer.load(tiny_tokenizer_path)
    p = tmp_path / "sft.jsonl"
    p.write_text(
        json.dumps({"instruction": "Q one", "response": "A one"})
        + "\n"
        + json.dumps({"instruction": "Q two", "response": "A two"})
        + "\n"
    )
    ex = load_sft_examples(p, tok, max_seq_len=128)
    assert len(ex) == 2


def test_sft_batches_pads_correctly(tiny_tokenizer_path):
    tok = KeithTokenizer.load(tiny_tokenizer_path)
    ex = [([1, 2, 3], [-1, 3, tok.eos_id]), ([1, 2], [-1, tok.eos_id])]
    rng = np.random.default_rng(0)
    ((x, y),) = list(sft_batches(ex, batch_size=2, pad_id=tok.pad_id, rng=rng, shuffle=False))
    assert x.shape == y.shape == (2, 3)  # padded to longest (3)
    assert x[1, 2] == tok.pad_id  # shorter row padded in input
    assert y[1, 2] == -1  # ...and masked in target (no loss on pad)


# --- SFTTrainer end to end ---


def _make_base_ckpt(path, tok, tmp_path):
    cfg = ModelConfig(
        vocab_size=tok.vocab_size, d_model=32, n_layers=2, n_heads=2, ffn_hidden=64, max_seq_len=128
    )
    torch.manual_seed(0)
    model = Transformer(cfg)
    opt = torch.optim.AdamW(model.parameters())
    save_checkpoint(path, model, opt, 0, cfg, TrainConfig(), np.random.default_rng(0))
    return cfg


def test_sft_trainer_end_to_end(tiny_tokenizer_path, tmp_path):
    tok = KeithTokenizer.load(tiny_tokenizer_path)
    base = tmp_path / "base.pt"
    cfg = _make_base_ckpt(base, tok, tmp_path)

    data = tmp_path / "sft.jsonl"
    data.write_text(
        "".join(
            json.dumps({"instruction": f"Question {i} about goblins?", "response": f"Answer {i}."})
            + "\n"
            for i in range(16)
        )
    )
    out = tmp_path / "sft_out"
    trainer = SFTTrainer(
        base_ckpt=base,
        data_jsonl=data,
        tokenizer_path=tiny_tokenizer_path,
        out_dir=out,
        epochs=5,
        lr=3e-3,
        batch_size=4,
        device="cpu",
        dtype="float32",
    )
    trainer.train()

    # checkpoint saved in the standard format (export-compatible)
    assert (out / "latest.pt").exists()
    ckpt = load_checkpoint(out / "latest.pt")
    assert ckpt["model_cfg"]["vocab_size"] == cfg.vocab_size
    assert "model_state" in ckpt
    # loss should drop over 5 epochs of overfitting a tiny set
    lines = [json.loads(ln) for ln in (out / "metrics.jsonl").read_text().splitlines()]
    assert lines[-1]["loss"] < lines[0]["loss"]


def test_sft_trainer_rejects_vocab_mismatch(tiny_tokenizer_path, tmp_path):
    import pytest

    tok = KeithTokenizer.load(tiny_tokenizer_path)
    base = tmp_path / "base.pt"
    # base with a different vocab than the tokenizer
    cfg = ModelConfig(vocab_size=tok.vocab_size + 10, d_model=32, n_layers=1, n_heads=2)
    torch.manual_seed(0)
    model = Transformer(cfg)
    save_checkpoint(
        base,
        model,
        torch.optim.AdamW(model.parameters()),
        0,
        cfg,
        TrainConfig(),
        np.random.default_rng(0),
    )
    data = tmp_path / "sft.jsonl"
    data.write_text(json.dumps({"instruction": "q", "response": "a"}) + "\n")
    with pytest.raises(ValueError, match="vocab"):
        SFTTrainer(
            base_ckpt=base,
            data_jsonl=data,
            tokenizer_path=tiny_tokenizer_path,
            out_dir=tmp_path / "o",
            device="cpu",
        )
