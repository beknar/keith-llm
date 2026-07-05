import pytest
import torch

from keith_llm.config import ModelConfig
from keith_llm.generate import _apply_repetition_penalty, generate, sample_token
from keith_llm.model import Transformer


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=64, d_model=32, n_layers=2, n_heads=2, ffn_hidden=64, max_seq_len=32
    )
    return Transformer(cfg).eval()


def test_temperature_zero_is_greedy():
    logits = torch.tensor([0.1, 3.0, -1.0, 2.9])
    assert sample_token(logits, temperature=0.0) == 1


def test_top_k_one_is_greedy():
    logits = torch.randn(100)
    for _ in range(5):
        assert sample_token(logits, temperature=1.0, top_k=1, top_p=1.0) == int(logits.argmax())


def test_top_p_keeps_dominant_token():
    logits = torch.tensor([10.0, 1.0, 0.5, 0.0])
    for _ in range(5):
        assert sample_token(logits, temperature=1.0, top_p=0.5) == 0


def test_top_p_never_empties_distribution():
    logits = torch.tensor([1.0, 1.0, 1.0, 1.0])
    assert sample_token(logits, temperature=1.0, top_p=0.01) in range(4)


def test_repetition_penalty_reduces_seen_positive_logits():
    logits = torch.tensor([2.0, 4.0, -2.0])
    out = _apply_repetition_penalty(logits.clone(), seen_ids=[1, 2], penalty=2.0)
    assert out[0] == 2.0  # unseen untouched
    assert out[1] == 2.0  # positive halved
    assert out[2] == -4.0  # negative pushed further down


def test_generate_deterministic_with_seed(model):
    prompt = [1, 2, 3]
    outs = []
    for _ in range(2):
        g = torch.Generator().manual_seed(42)
        outs.append(generate(model, prompt, max_new_tokens=10, generator=g))
    assert outs[0] == outs[1]
    assert outs[0][:3] == prompt
    assert len(outs[0]) <= 3 + 10


def test_generate_stops_on_stop_id(model):
    out = generate(model, [1, 2], max_new_tokens=20, stop_ids=range(64))
    assert len(out) == 3  # every token is a stop token -> exactly one generated


def test_generate_respects_max_seq_len(model):
    out = generate(model, [1] * 30, max_new_tokens=50, temperature=0.0)
    assert len(out) <= model.cfg.max_seq_len


def test_generate_rejects_empty_prompt(model):
    with pytest.raises(ValueError, match="empty"):
        generate(model, [], max_new_tokens=5)


def test_generate_rejects_overlong_prompt(model):
    with pytest.raises(ValueError, match="max_seq_len"):
        generate(model, [1] * 32, max_new_tokens=5)


def test_generate_frees_cache_and_restores_mode(model):
    model.train()
    try:
        generate(model, [1, 2], max_new_tokens=3)
        assert model.training
        assert model.blocks[0].attention.cache_k is None
    finally:
        model.eval()
