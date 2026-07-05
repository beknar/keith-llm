import shutil

import gguf
import numpy as np
import pytest
import torch

from keith_llm.config import ModelConfig, TrainConfig
from keith_llm.export.gguf_export import export_gguf
from keith_llm.export.ollama import write_modelfile
from keith_llm.export.quantize import quantize
from keith_llm.model import Transformer
from keith_llm.tokenizer.wrapper import SPECIAL_TOKENS, KeithTokenizer
from keith_llm.train.checkpoint import save_checkpoint


@pytest.fixture(scope="module")
def exported(tiny_tokenizer_path, tmp_path_factory):
    """Tiny random-init model saved as a checkpoint, exported to GGUF, and
    opened with GGUFReader. Returns (cfg, tokenizer, reader, gguf_path)."""
    tmp = tmp_path_factory.mktemp("export")
    tok = KeithTokenizer.load(tiny_tokenizer_path)
    cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        d_model=32,
        n_layers=2,
        n_heads=2,
        ffn_hidden=64,
        max_seq_len=64,
    )
    torch.manual_seed(0)
    model = Transformer(cfg)
    optimizer = torch.optim.AdamW(model.parameters())
    ckpt_path = tmp / "latest.pt"
    save_checkpoint(ckpt_path, model, optimizer, 0, cfg, TrainConfig(), np.random.default_rng(0))
    gguf_path = export_gguf(ckpt_path, tiny_tokenizer_path, tmp / "model-f16.gguf")
    return cfg, tok, gguf.GGUFReader(gguf_path), gguf_path


def _field(reader, key):
    field = reader.get_field(key)
    assert field is not None, f"missing GGUF field {key}"
    return field.contents()


def test_architecture_fields(exported):
    cfg, _, reader, _ = exported
    assert _field(reader, "general.architecture") == "llama"
    assert _field(reader, "llama.context_length") == cfg.max_seq_len
    assert _field(reader, "llama.embedding_length") == cfg.d_model
    assert _field(reader, "llama.block_count") == cfg.n_layers
    assert _field(reader, "llama.feed_forward_length") == cfg.ffn_hidden
    assert _field(reader, "llama.attention.head_count") == cfg.n_heads
    assert _field(reader, "llama.attention.head_count_kv") == cfg.n_heads
    assert _field(reader, "llama.attention.layer_norm_rms_epsilon") == pytest.approx(cfg.norm_eps)
    assert _field(reader, "llama.rope.freq_base") == pytest.approx(cfg.rope_theta)
    assert _field(reader, "llama.rope.dimension_count") == cfg.head_dim
    assert _field(reader, "llama.vocab_size") == cfg.vocab_size


def test_tokenizer_fields(exported):
    cfg, tok, reader, _ = exported
    assert _field(reader, "tokenizer.ggml.model") == "gpt2"
    assert _field(reader, "tokenizer.ggml.pre") == "gpt-2"
    tokens = _field(reader, "tokenizer.ggml.tokens")
    types = _field(reader, "tokenizer.ggml.token_type")
    assert len(tokens) == len(types) == cfg.vocab_size
    assert len(_field(reader, "tokenizer.ggml.merges")) > 0
    assert _field(reader, "tokenizer.ggml.bos_token_id") == tok.bos_id
    assert _field(reader, "tokenizer.ggml.eos_token_id") == tok.eos_id
    assert _field(reader, "tokenizer.ggml.padding_token_id") == tok.pad_id
    assert _field(reader, "tokenizer.ggml.add_bos_token") is True

    for special in SPECIAL_TOKENS:
        tid = tok.token_id(special)
        assert tokens[tid] == special
        assert types[tid] == int(gguf.TokenType.CONTROL), special


def test_tensor_set_complete(exported):
    cfg, _, reader, _ = exported
    names = {t.name for t in reader.tensors}
    expected = {"token_embd.weight", "output_norm.weight", "output.weight"}
    for i in range(cfg.n_layers):
        expected |= {
            f"blk.{i}.{suffix}"
            for suffix in (
                "attn_norm.weight",
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "ffn_norm.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            )
        }
    assert names == expected


def test_tensor_shapes_and_dtypes(exported):
    cfg, _, reader, _ = exported
    by_name = {t.name: t for t in reader.tensors}
    emb = by_name["token_embd.weight"]
    # GGUF stores dims reversed (ggml order).
    assert sorted(emb.shape.tolist()) == sorted([cfg.vocab_size, cfg.d_model])
    assert emb.tensor_type == gguf.GGMLQuantizationType.F16
    assert by_name["output_norm.weight"].tensor_type == gguf.GGMLQuantizationType.F32
    gate = by_name["blk.0.ffn_gate.weight"]
    assert sorted(gate.shape.tolist()) == sorted([cfg.ffn_hidden, cfg.d_model])


def test_embedding_values_roundtrip(exported, tmp_path):
    cfg, _, reader, gguf_path = exported
    from keith_llm.train.checkpoint import load_checkpoint

    ckpt = load_checkpoint(gguf_path.parent / "latest.pt")
    original = ckpt["model_state"]["tok_emb.weight"].numpy().astype(np.float16)
    stored = {t.name: t for t in reader.tensors}["token_embd.weight"].data
    assert np.array_equal(stored.reshape(original.shape), original)


def test_export_rejects_oversized_tokenizer(exported, tiny_tokenizer_path, tmp_path):
    cfg, tok, _, gguf_path = exported
    small_cfg_ckpt = gguf_path.parent / "latest.pt"
    from keith_llm.train.checkpoint import load_checkpoint

    ckpt = load_checkpoint(small_cfg_ckpt)
    ckpt["model_cfg"]["vocab_size"] = tok.vocab_size - 1
    bad = tmp_path / "bad.pt"
    torch.save(ckpt, bad)
    with pytest.raises(ValueError, match="exceeds"):
        export_gguf(bad, tiny_tokenizer_path, tmp_path / "bad.gguf")


def test_modelfile_contents(exported, tmp_path):
    _, _, _, gguf_path = exported
    modelfile = write_modelfile(gguf_path, tmp_path / "Modelfile")
    text = modelfile.read_text()
    assert f"FROM ./{gguf_path.name}" in text
    assert 'TEMPLATE "{{ .Prompt }}"' in text
    assert "PARAMETER stop <|eos|>" in text


def test_modelfile_missing_gguf(tmp_path):
    with pytest.raises(FileNotFoundError):
        write_modelfile(tmp_path / "nope.gguf")


def test_quantize_rejects_bad_qtype(exported):
    _, _, _, gguf_path = exported
    with pytest.raises(ValueError, match="qtype"):
        quantize(gguf_path, "Q2_K")


def test_quantize_missing_binary(exported, monkeypatch):
    _, _, _, gguf_path = exported
    monkeypatch.setenv("LLAMA_QUANTIZE", "/nonexistent/llama-quantize")
    with pytest.raises(FileNotFoundError, match="llama-quantize"):
        quantize(gguf_path, "Q8_0")


@pytest.mark.skipif(shutil.which("llama-quantize") is None, reason="llama.cpp not installed")
def test_quantize_q8_0(exported, tmp_path):
    _, _, _, gguf_path = exported
    out = quantize(gguf_path, "Q8_0", out_path=tmp_path / "model-Q8_0.gguf")
    assert out.exists() and out.stat().st_size > 0
