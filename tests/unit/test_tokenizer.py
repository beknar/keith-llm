import pytest

from keith_llm.tokenizer.wrapper import SPECIAL_TOKENS, KeithTokenizer


@pytest.fixture(scope="module")
def tok(tiny_tokenizer_path):
    return KeithTokenizer.load(tiny_tokenizer_path)


def test_vocab_size_within_budget(tok):
    # The tiny corpus saturates before 512 merges; a real corpus reaches the
    # requested size exactly. The hard requirements are the ceiling and that
    # the byte alphabet + specials are all present.
    assert 256 + len(SPECIAL_TOKENS) < tok.vocab_size <= 512


def test_roundtrip_byte_exact(tok):
    text = "The gnoll «Yeenoghu» rolls 2d6+3 — “fire” damage! (naïve save ﬂubbed)"
    assert tok.decode(tok.encode(text)) == text


def test_roundtrip_plain_prose(tok):
    text = "The goblin ambushes the caravan at the edge of the mirewood."
    assert tok.decode(tok.encode(text)) == text


def test_special_tokens_are_single_ids(tok):
    for t in SPECIAL_TOKENS:
        ids = tok.encode(t)
        assert ids == [tok.token_id(t)], f"{t} split into {len(ids)} tokens"


def test_special_tokens_atomic_inside_text(tok):
    ids = tok.encode("<|system:dnd5e|>The goblin waits.")
    assert ids[0] == tok.token_id("<|system:dnd5e|>")
    assert tok.decode(ids) == "The goblin waits."  # specials skipped on decode


def test_specials_occupy_top_ids(tok):
    floor = tok.vocab_size - len(SPECIAL_TOKENS)
    for t in SPECIAL_TOKENS:
        assert tok.token_id(t) >= floor


def test_control_prefix(tok):
    prefix = tok.control_prefix("dnd5e", "adventure")
    assert prefix == [
        tok.bos_id,
        tok.token_id("<|system:dnd5e|>"),
        tok.token_id("<|doc:adventure|>"),
    ]


def test_control_prefix_rejects_unknown(tok):
    with pytest.raises(ValueError, match="system"):
        tok.control_prefix("gurps", "adventure")
    with pytest.raises(ValueError, match="doc_type"):
        tok.control_prefix("dnd5e", "screenplay")


def test_save_load_equivalence(tok, tmp_path):
    out = tmp_path / "tok.json"
    tok.save(out)
    reloaded = KeithTokenizer.load(out)
    text = "A wyvern circles the sunken temple, rolling 4d8 cold damage."
    assert reloaded.encode(text) == tok.encode(text)
    assert reloaded.vocab_size == tok.vocab_size


def test_load_rejects_tokenizer_without_specials(tmp_path):
    from tokenizers import Tokenizer, models

    bare = Tokenizer(models.BPE(unk_token=None))
    p = tmp_path / "bare.json"
    bare.save(str(p))
    with pytest.raises(ValueError, match="special"):
        KeithTokenizer.load(p)
