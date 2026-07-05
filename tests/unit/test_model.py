from pathlib import Path

import pytest
import torch

from keith_llm.config import ModelConfig, load_config
from keith_llm.model import Transformer, apply_rotary, precompute_freqs_cis

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

NOMINAL_PARAMS = {
    "tiny-1m": 1.2e6,
    "25m": 29e6,
    "125m": 126e6,
    "350m": 352e6,
    "500m": 486e6,
}


@pytest.fixture(scope="module")
def cfg():
    return ModelConfig(
        vocab_size=64, d_model=32, n_layers=2, n_heads=2, ffn_hidden=64, max_seq_len=32
    )


@pytest.fixture(scope="module")
def model(cfg):
    torch.manual_seed(0)
    return Transformer(cfg).eval()


def test_rope_preserves_norm():
    freqs = precompute_freqs_cis(16, 32, 10000.0)
    x = torch.randn(2, 32, 4, 16)
    rotated = apply_rotary(x, freqs)
    assert torch.allclose(x.norm(dim=-1), rotated.norm(dim=-1), atol=1e-5)


def test_rope_depends_only_on_relative_position():
    freqs = precompute_freqs_cis(16, 64, 10000.0)
    q_vec = torch.randn(16)
    k_vec = torch.randn(16)
    q = apply_rotary(q_vec.view(1, 1, 1, 16).repeat(1, 64, 1, 1), freqs)
    k = apply_rotary(k_vec.view(1, 1, 1, 16).repeat(1, 64, 1, 1), freqs)
    dot_early = (q[0, 0, 0] * k[0, 4, 0]).sum()
    dot_late = (q[0, 37, 0] * k[0, 41, 0]).sum()
    assert torch.allclose(dot_early, dot_late, atol=1e-4)


def test_rope_position_zero_is_identity():
    freqs = precompute_freqs_cis(16, 8, 10000.0)
    x = torch.randn(1, 1, 2, 16)
    assert torch.allclose(apply_rotary(x, freqs[:1]), x, atol=1e-6)


def test_causality(model, cfg):
    torch.manual_seed(1)
    idx = torch.randint(0, cfg.vocab_size, (1, 16))
    logits_a, _ = model(idx)
    perturbed = idx.clone()
    perturbed[0, -1] = (perturbed[0, -1] + 1) % cfg.vocab_size
    logits_b, _ = model(perturbed)
    assert torch.allclose(logits_a[0, :-1], logits_b[0, :-1], atol=1e-5)
    assert not torch.allclose(logits_a[0, -1], logits_b[0, -1], atol=1e-3)


def test_kv_cache_prefill_matches_plain_forward(model, cfg):
    torch.manual_seed(2)
    idx = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        plain, _ = model(idx)
        model.allocate_kv_cache(1, device="cpu")
        cached, _ = model(idx, start_pos=0)
        model.free_kv_cache()
    assert torch.allclose(plain, cached, atol=1e-4)


def test_kv_cache_stepwise_decode_matches_plain_forward(model, cfg):
    torch.manual_seed(3)
    idx = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        plain, _ = model(idx)
        model.allocate_kv_cache(1, device="cpu")
        stepwise = [model(idx[:, :8], start_pos=0)[0][0, -1]]
        for pos in range(8, 16):
            stepwise.append(model(idx[:, pos : pos + 1], start_pos=pos)[0][0, -1])
        model.free_kv_cache()
    for i, step_logits in enumerate(stepwise):
        assert torch.allclose(plain[0, 7 + i], step_logits, atol=1e-4), f"pos {7 + i}"


def test_multi_token_forward_mid_cache_rejected(model, cfg):
    model.allocate_kv_cache(1, device="cpu")
    try:
        with pytest.raises(ValueError, match="start_pos"):
            model(torch.zeros(1, 4, dtype=torch.long), start_pos=2)
    finally:
        model.free_kv_cache()


def test_forward_without_cache_allocation_rejected(model):
    with pytest.raises(RuntimeError, match="cache"):
        model(torch.zeros(1, 4, dtype=torch.long), start_pos=0)


def test_sequence_length_limit(model, cfg):
    with pytest.raises(ValueError, match="max_seq_len"):
        model(torch.zeros(1, cfg.max_seq_len + 1, dtype=torch.long))


def test_loss_computed_with_targets(model, cfg):
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    _, loss = model(idx, targets=idx)
    assert loss is not None and loss.item() > 0


def test_tied_embeddings(model):
    assert model.lm_head.weight is model.tok_emb.weight


def test_untied_embeddings_are_separate():
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=2, tie_embeddings=False)
    m = Transformer(cfg)
    assert m.lm_head.weight is not m.tok_emb.weight


@pytest.mark.parametrize("preset", sorted(NOMINAL_PARAMS))
def test_preset_param_counts(preset):
    model_cfg, _ = load_config(CONFIG_DIR / f"{preset}.yaml")
    with torch.device("meta"):
        m = Transformer(model_cfg)
    nominal = NOMINAL_PARAMS[preset]
    assert abs(m.num_params() - nominal) / nominal < 0.06, (
        f"{preset}: {m.num_params():,} vs nominal {nominal:,.0f}"
    )
