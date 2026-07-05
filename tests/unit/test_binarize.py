import json

import numpy as np
import pytest

from keith_llm.data.binarize import binarize
from keith_llm.tokenizer.wrapper import KeithTokenizer


@pytest.fixture(scope="module")
def bins(tiny_corpus, tiny_tokenizer_path, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("tokens")
    meta = binarize(tiny_corpus, tiny_tokenizer_path, out_dir, val_mod=5)
    return out_dir, meta


def test_meta_fields(bins, tiny_tokenizer_path):
    out_dir, meta = bins
    assert meta["dtype"] == "uint16"
    assert meta["vocab_size"] == KeithTokenizer.load(tiny_tokenizer_path).vocab_size
    assert meta["n_train_tokens"] > 0
    assert meta["n_val_tokens"] > 0
    assert meta["n_documents"] == 36
    assert len(meta["tokenizer_sha1"]) == 40
    assert json.loads((out_dir / "meta.json").read_text()) == meta


def test_bin_sizes_match_meta(bins):
    out_dir, meta = bins
    train = np.fromfile(out_dir / "train.bin", dtype=np.uint16)
    val = np.fromfile(out_dir / "val.bin", dtype=np.uint16)
    assert len(train) == meta["n_train_tokens"]
    assert len(val) == meta["n_val_tokens"]
    assert train.max() < meta["vocab_size"]


def test_first_document_structure(bins, tiny_corpus, tiny_tokenizer_path):
    out_dir, _ = bins
    tok = KeithTokenizer.load(tiny_tokenizer_path)
    records = [json.loads(line) for line in open(tiny_corpus)]
    stream = np.fromfile(out_dir / "train.bin", dtype=np.uint16)

    # First train-split document starts at offset 0 with its control prefix.
    first_train = next(r for r in records if _split_of(r, records) == "train")
    prefix = tok.control_prefix(first_train["system"], first_train["doc_type"])
    assert list(stream[:3]) == prefix
    doc_len = len(prefix) + len(tok.encode(first_train["text"])) + 1
    assert stream[doc_len - 1] == tok.eos_id
    decoded = tok.decode(stream[:doc_len].tolist())
    assert decoded == first_train["text"]


def _split_of(rec, records):
    # Mirrors binarize's val logic for val_mod=5, including the promotion rule.
    val = {r["id"] for r in records if int(r["id"][:8], 16) % 5 == 0}
    if not val and len(records) > 1:
        val = {min(records, key=lambda r: (len(r["text"]), r["id"]))["id"]}
    return "val" if rec["id"] in val else "train"


def test_val_promotion_when_nothing_hashes_to_val(tiny_corpus, tiny_tokenizer_path, tmp_path):
    records = [json.loads(line) for line in open(tiny_corpus)][:2]
    # Pick val_mod so neither doc hashes into val naturally.
    val_mod = 7919
    while any(int(r["id"][:8], 16) % val_mod == 0 for r in records):
        val_mod += 1
    corpus = tmp_path / "two.jsonl"
    corpus.write_text("".join(json.dumps(r) + "\n" for r in records))
    meta = binarize(corpus, tiny_tokenizer_path, tmp_path / "bins", val_mod=val_mod)
    assert meta["n_val_documents"] == 1
    assert meta["n_val_tokens"] > 0
    # The smaller document is the one promoted.
    assert meta["n_val_tokens"] <= meta["n_train_tokens"]


def test_empty_corpus_rejected(tiny_tokenizer_path, tmp_path):
    corpus = tmp_path / "empty.jsonl"
    corpus.write_text("")
    with pytest.raises(ValueError, match="empty"):
        binarize(corpus, tiny_tokenizer_path, tmp_path / "bins")


def test_binarize_deterministic(tiny_corpus, tiny_tokenizer_path, tmp_path):
    m1 = binarize(tiny_corpus, tiny_tokenizer_path, tmp_path / "a", val_mod=5)
    m2 = binarize(tiny_corpus, tiny_tokenizer_path, tmp_path / "b", val_mod=5)
    assert m1 == m2
    assert (tmp_path / "a" / "train.bin").read_bytes() == (
        tmp_path / "b" / "train.bin"
    ).read_bytes()
