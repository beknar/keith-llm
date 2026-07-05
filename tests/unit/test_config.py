from pathlib import Path

import pytest

from keith_llm.config import ModelConfig, TrainConfig, load_config

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
PRESETS = sorted(CONFIG_DIR.glob("*.yaml"))


def test_presets_exist():
    names = {p.stem for p in PRESETS}
    assert {"tiny-1m", "25m", "125m", "350m", "500m"} <= names


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.stem)
def test_preset_loads(preset):
    model, train = load_config(preset)
    assert model.vocab_size == 16384
    assert model.d_model % model.n_heads == 0
    assert train.max_steps > 0
    assert train.dtype in ("float32", "bfloat16")


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.stem)
def test_gpu_presets_kquant_compatible(preset):
    # llama.cpp k-quants need tensor row sizes divisible by 256; tiny-1m is
    # exempt (test-only, Q8_0 path).
    model, _ = load_config(preset)
    if preset.stem == "tiny-1m":
        return
    assert model.d_model % 256 == 0
    assert model.ffn_hidden % 256 == 0


def test_head_dim():
    assert ModelConfig(d_model=768, n_heads=12).head_dim == 64


def test_rejects_indivisible_heads():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(d_model=100, n_heads=3)


def test_rejects_oversized_vocab():
    with pytest.raises(ValueError, match="uint16"):
        ModelConfig(vocab_size=70000)


def test_rejects_unknown_keys(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("model:\n  d_modle: 64\n")
    with pytest.raises(ValueError, match="d_modle"):
        load_config(cfg)


def test_rejects_unknown_section(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("modle:\n  d_model: 64\n")
    with pytest.raises(ValueError, match="modle"):
        load_config(cfg)


def test_rejects_bad_dtype():
    with pytest.raises(ValueError, match="dtype"):
        TrainConfig(dtype="float16")


def test_rejects_warmup_exceeding_max_steps():
    with pytest.raises(ValueError, match="warmup"):
        TrainConfig(warmup_steps=500, max_steps=100)


def test_empty_config_uses_defaults(tmp_path):
    cfg = tmp_path / "empty.yaml"
    cfg.write_text("")
    model, train = load_config(cfg)
    assert model.d_model == ModelConfig().d_model
    assert train.lr == TrainConfig().lr
