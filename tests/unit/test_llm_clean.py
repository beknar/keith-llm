import json

from keith_llm.data.audit import score_text
from keith_llm.data.llm_clean import (
    _chunks,
    _strip_fences,
    clean_corpus,
    clean_text,
    faithfulness,
)

# A document that extracts badly: run-together words (internal caps) trip the
# audit BAD verdict. Its clean form uses the same letters, just re-spaced.
BAD_TEXT = " ".join(["theGoblinattacksTheHeroWithaScimitar"] * 40)
GOOD_TEXT = " ".join(["the goblin attacks the hero with a scimitar."] * 40)


class _FakeClient:
    """Returns a canned reply per chat call; records prompts."""

    def __init__(self, reply, boom=False):
        self.reply = reply
        self.boom = boom
        self.prompts = []

    def chat(self, user, system=None):
        self.prompts.append(user)
        if self.boom:
            raise RuntimeError("model down")
        return self.reply


# --- faithfulness ---


def test_faithfulness_high_when_only_respacing():
    # same letters, different spacing -> containment ~ 1.0
    assert faithfulness("theGoblinAttacks", "the Goblin Attacks") > 0.95


def test_faithfulness_low_when_invented():
    # rewrite about a totally different subject shares few letter 4-grams
    assert faithfulness("the goblin attacks", "quantum photons xylophone zzz") < 0.3


def test_faithfulness_zero_on_empty_clean():
    assert faithfulness("anything", "") == 0.0


# --- chunking ---


def test_chunks_small_text_single():
    assert _chunks("hello world", 6000) == ["hello world"]


def test_chunks_respects_budget_and_covers_all():
    text = "\n\n".join(f"paragraph number {i} with some words" for i in range(200))
    chunks = _chunks(text, 200)
    assert all(len(c) <= 200 for c in chunks)
    # every paragraph survives somewhere in the concatenation
    assert "paragraph number 199" in "".join(chunks)


def test_chunks_hard_splits_oversize_paragraph():
    big = "x" * 5000  # one paragraph, no blank lines, over budget
    chunks = _chunks(big, 1000)
    assert len(chunks) >= 5
    assert all(len(c) <= 1000 for c in chunks)
    # every character is preserved (a paragraph separator may be appended)
    assert "".join(chunks).rstrip("\n") == big


# --- strip fences ---


def test_strip_fences_removes_wrapping():
    assert _strip_fences("```\nhello\n```") == "hello"
    assert _strip_fences("```text\nhello\nworld\n```") == "hello\nworld"
    assert _strip_fences("no fence here") == "no fence here"


# --- clean_text ---


def test_clean_text_chunks_and_joins():
    client = _FakeClient("REPAIRED")
    text = "\n\n".join(f"para {i} " * 50 for i in range(10))
    out = clean_text(client, text, max_chars=300)
    assert len(client.prompts) > 1  # actually chunked
    assert out == "\n\n".join(["REPAIRED"] * len(client.prompts))
    # the constrained instructions reached the model
    assert "Do NOT summarize" in client.prompts[0]


# --- clean_corpus integration ---


def _write_corpus(tmp_path, records):
    p = tmp_path / "corpus.jsonl"
    with p.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return p


def test_bad_doc_replaced_when_improved_and_faithful(tmp_path):
    assert score_text(BAD_TEXT)["verdict"] == "BAD"
    corpus = _write_corpus(
        tmp_path,
        [
            {"source": "bad.pdf", "system": "dnd5e", "doc_type": "rules", "text": BAD_TEXT},
            {"source": "ok.pdf", "system": "dnd5e", "doc_type": "rules", "text": GOOD_TEXT},
        ],
    )
    out = tmp_path / "cleaned.jsonl"
    client = _FakeClient(GOOD_TEXT)
    stats = clean_corpus(corpus, out, client=client)

    assert stats["targeted"] == 1  # only the BAD doc was sent to the model
    assert stats["replaced"] == 1
    assert len(client.prompts) == 1  # the OK doc was never touched
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    bad = next(r for r in recs if r["source"] == "bad.pdf")
    assert bad["cleaned"] is True
    assert bad["text"] == GOOD_TEXT
    assert bad["id"] == __import__("hashlib").sha1(GOOD_TEXT.encode()).hexdigest()


def test_unfaithful_rewrite_is_rejected(tmp_path):
    corpus = _write_corpus(
        tmp_path, [{"source": "bad.pdf", "system": "dnd5e", "doc_type": "rules", "text": BAD_TEXT}]
    )
    out = tmp_path / "cleaned.jsonl"
    # model "cleans" it into readable prose about something else entirely
    client = _FakeClient(" ".join(["quantum photon lattice resonance cascade"] * 40))
    stats = clean_corpus(corpus, out, client=client)

    assert stats["replaced"] == 0
    assert stats["unfaithful"] == 1
    assert stats["kept"] == 1
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    assert recs[0]["text"] == BAD_TEXT  # original preserved


def test_drop_failed_drops_unfixable_bad(tmp_path):
    corpus = _write_corpus(
        tmp_path, [{"source": "bad.pdf", "system": "dnd5e", "doc_type": "rules", "text": BAD_TEXT}]
    )
    out = tmp_path / "cleaned.jsonl"
    client = _FakeClient(" ".join(["quantum photon lattice resonance"] * 40))  # unfaithful
    stats = clean_corpus(corpus, out, client=client, drop_failed=True)

    assert stats["dropped"] == 1
    assert stats["documents_out"] == 0
    assert out.read_text() == ""


def test_clean_failure_keeps_original(tmp_path):
    corpus = _write_corpus(
        tmp_path, [{"source": "bad.pdf", "system": "dnd5e", "doc_type": "rules", "text": BAD_TEXT}]
    )
    out = tmp_path / "cleaned.jsonl"
    stats = clean_corpus(corpus, out, client=_FakeClient("", boom=True))
    assert stats["kept"] == 1
    assert stats["replaced"] == 0
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    assert recs[0]["text"] == BAD_TEXT


def test_bad_only_skips_warn(tmp_path, monkeypatch):
    import keith_llm.data.llm_clean as lc

    # force one doc WARN, one BAD regardless of exact heuristics
    verdicts = iter(["WARN", "BAD"])
    real = lc.score_text

    def fake_score(text):
        m = dict(real(text))
        try:
            m["verdict"] = next(verdicts)
        except StopIteration:
            pass
        return m

    monkeypatch.setattr(lc, "score_text", fake_score)
    corpus = _write_corpus(
        tmp_path,
        [
            {"source": "warn.pdf", "system": "d", "doc_type": "r", "text": BAD_TEXT},
            {"source": "bad.pdf", "system": "d", "doc_type": "r", "text": BAD_TEXT},
        ],
    )
    out = tmp_path / "cleaned.jsonl"
    client = _FakeClient(GOOD_TEXT)
    stats = clean_corpus(corpus, out, client=client, target_verdicts=("BAD",))
    assert stats["targeted"] == 1  # only the BAD doc


def test_max_docs_caps_processing(tmp_path):
    corpus = _write_corpus(
        tmp_path,
        [
            {"source": f"bad{i}.pdf", "system": "d", "doc_type": "r", "text": BAD_TEXT}
            for i in range(3)
        ],
    )
    out = tmp_path / "cleaned.jsonl"
    client = _FakeClient(GOOD_TEXT)
    stats = clean_corpus(corpus, out, client=client, max_docs=1)
    assert stats["targeted"] == 1
    assert stats["documents_out"] == 3  # the two over the cap pass through untouched


def test_dry_run_writes_nothing(tmp_path):
    corpus = _write_corpus(
        tmp_path, [{"source": "bad.pdf", "system": "d", "doc_type": "r", "text": BAD_TEXT}]
    )
    out = tmp_path / "cleaned.jsonl"
    stats = clean_corpus(corpus, out, client=_FakeClient(GOOD_TEXT), dry_run=True)
    assert stats["replaced"] == 1
    assert stats["dry_run"] is True
    assert not out.exists()


def test_unreachable_ollama_errors(tmp_path):
    import pytest

    corpus = _write_corpus(
        tmp_path, [{"source": "bad.pdf", "system": "d", "doc_type": "r", "text": BAD_TEXT}]
    )
    with pytest.raises(SystemExit, match="ollama not reachable"):
        clean_corpus(corpus, tmp_path / "o.jsonl", ollama_url="http://127.0.0.1:1")
