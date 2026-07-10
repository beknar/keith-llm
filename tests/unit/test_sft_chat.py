import builtins

import torch

from keith_llm.config import ModelConfig
from keith_llm.export.ollama import write_modelfile
from keith_llm.model import Transformer
from keith_llm.sft.chat import chat_once, chat_repl
from keith_llm.sft.format import INSTRUCTION_HEADER, RESPONSE_HEADER
from keith_llm.tokenizer.wrapper import KeithTokenizer

# --- chat Modelfile ---


def _gguf(tmp_path):
    g = tmp_path / "m.gguf"
    g.write_bytes(b"GGUF")
    return g


def test_chat_modelfile_matches_training_template(tmp_path):
    mf = write_modelfile(_gguf(tmp_path), tmp_path / "Modelfile", chat=True)
    text = mf.read_text()
    # the rendered template must equal build_prompt with ollama's placeholder
    assert f"{INSTRUCTION_HEADER}{{{{ .Prompt }}}}{RESPONSE_HEADER}" in text
    assert 'PARAMETER stop "<|eos|>"' in text
    assert 'PARAMETER stop "### Instruction:"' in text
    assert "PARAMETER num_predict" in text


def test_completion_modelfile_still_raw(tmp_path):
    text = write_modelfile(_gguf(tmp_path), tmp_path / "Modelfile", chat=False).read_text()
    assert 'TEMPLATE "{{ .Prompt }}"' in text
    assert "### Instruction:" not in text


# --- chat_once ---


def _tiny_model(tok):
    cfg = ModelConfig(
        vocab_size=tok.vocab_size, d_model=32, n_layers=2, n_heads=2, ffn_hidden=64, max_seq_len=128
    )
    torch.manual_seed(0)
    return Transformer(cfg).eval()


def test_chat_once_returns_response_only(tiny_tokenizer_path):
    tok = KeithTokenizer.load(tiny_tokenizer_path)
    model = _tiny_model(tok)
    g = torch.Generator().manual_seed(0)
    reply = chat_once(model, tok, "What is a goblin?", max_new_tokens=16, generator=g)
    assert isinstance(reply, str)
    # response is decoded generation only — it must not echo the instruction prompt
    assert "### Instruction:" not in reply
    assert "### Response:" not in reply


def test_chat_once_deterministic(tiny_tokenizer_path):
    tok = KeithTokenizer.load(tiny_tokenizer_path)
    model = _tiny_model(tok)
    a = chat_once(model, tok, "hi", max_new_tokens=12, generator=torch.Generator().manual_seed(1))
    b = chat_once(model, tok, "hi", max_new_tokens=12, generator=torch.Generator().manual_seed(1))
    assert a == b


def test_chat_repl_exits_and_answers(tiny_tokenizer_path, monkeypatch, capsys):
    tok = KeithTokenizer.load(tiny_tokenizer_path)
    model = _tiny_model(tok)
    replies = iter(["tell me about dragons", "exit"])
    monkeypatch.setattr(builtins, "input", lambda *a: next(replies))
    chat_repl(model, tok, max_new_tokens=8)
    out = capsys.readouterr().out
    assert "bot>" in out  # produced at least one reply before exiting


def test_chat_repl_stops_on_eof(tiny_tokenizer_path, monkeypatch):
    tok = KeithTokenizer.load(tiny_tokenizer_path)
    model = _tiny_model(tok)

    def eof(*a):
        raise EOFError

    monkeypatch.setattr(builtins, "input", eof)
    chat_repl(model, tok)  # must return, not hang or raise
