import json

from keith_llm.llm import OllamaClient
from keith_llm.sft.build import _sample_corpus_docs, build_sft_dataset
from keith_llm.sft.synth import parse_pairs, synth_corpus_pairs, synth_monster_pairs

MON = {
    "name": "Mire Goblin",
    "ac": [{"ac": 15}],
    "hp": {"average": 7, "formula": "2d6"},
    "cr": "1/4",
    "_copy": None,
}


# --- parse_pairs: robust JSON extraction ---


def test_parse_pairs_clean_json():
    r = '[{"question": "What is its AC?", "answer": "15."}, {"question": "HP?", "answer": "7."}]'
    assert parse_pairs(r) == [("What is its AC?", "15."), ("HP?", "7.")]


def test_parse_pairs_wrapped_in_prose_and_fences():
    r = 'Sure! Here you go:\n```json\n[{"question": "Q1", "answer": "A1"}]\n```\nHope that helps.'
    assert parse_pairs(r) == [("Q1", "A1")]


def test_parse_pairs_skips_malformed_items():
    r = '[{"question": "Q", "answer": "A"}, {"question": ""}, {"foo": "bar"}, "junk"]'
    assert parse_pairs(r) == [("Q", "A")]


def test_parse_pairs_no_json():
    assert parse_pairs("I could not do that.") == []
    assert parse_pairs("[not valid json]") == []


def test_parse_pairs_survives_stray_brackets_in_prose():
    # brackets before AND after the real array must not corrupt the span
    r = 'As requested [see below]:\n[{"question": "Q", "answer": "A"}]\nNote [1]: grounded.'
    assert parse_pairs(r) == [("Q", "A")]


def test_parse_pairs_takes_first_wellformed_array():
    # a leading non-array bracket, then two arrays -> first valid array wins
    r = '[oops\n[{"question": "Q1", "answer": "A1"}]\n[{"question": "Q2", "answer": "A2"}]'
    assert parse_pairs(r) == [("Q1", "A1")]


def test_parse_pairs_non_string_input():
    assert parse_pairs(None) == []


def test_parse_pairs_accepts_instruction_response_keys():
    r = '[{"instruction": "Explain initiative.", "response": "Turn order in a fight."}]'
    assert parse_pairs(r) == [("Explain initiative.", "Turn order in a fight.")]


def test_parse_pairs_null_value_is_not_stringified_to_None():
    # a JSON null value must be skipped, never become the literal string "None"
    r = '[{"instruction": null, "response": "R"}, {"instruction": "I", "response": "R2"}]'
    assert parse_pairs(r) == [("I", "R2")]


def test_parse_pairs_both_key_styles_and_missing_keys():
    # mixed key styles, plus an item with neither -> the empty one is skipped
    # (must not become the literal string "None")
    r = (
        '[{"instruction": "I1", "response": "R1"}, '
        '{"question": "Q2", "answer": "A2"}, {"foo": "bar"}]'
    )
    assert parse_pairs(r) == [("I1", "R1"), ("Q2", "A2")]


# --- synth_monster_pairs with a fake client ---


class _FakeClient:
    def __init__(self, response, boom=False):
        self.response = response
        self.boom = boom
        self.last_prompt = None

    def chat(self, user, system=None):
        self.last_prompt = user
        if self.boom:
            raise RuntimeError("model unavailable")
        return self.response


def test_synth_monster_pairs_grounds_on_json():
    client = _FakeClient('[{"question": "AC?", "answer": "15."}]')
    pairs = synth_monster_pairs(client, MON, n_pairs=3)
    assert pairs == [("AC?", "15.")]
    # the real monster JSON (minus private keys) is in the prompt -> grounded
    assert '"name": "Mire Goblin"' in client.last_prompt
    assert "_copy" not in client.last_prompt


def test_synth_failure_returns_empty_not_raise():
    assert synth_monster_pairs(_FakeClient("", boom=True), MON) == []


def test_synth_skips_copies_and_nameless():
    c = _FakeClient('[{"question":"Q","answer":"A"}]')
    assert synth_monster_pairs(c, {"name": "X", "_copy": {"n": 1}}) == []
    assert synth_monster_pairs(c, {"ac": [12]}) == []


# --- OllamaClient (no network) ---


def test_client_available_false_when_unreachable():
    # nothing listening on this port -> available() returns False, never raises
    assert OllamaClient("m", base_url="http://127.0.0.1:1").available() is False


class _FakeResp:
    def __init__(self, content):
        self._content = content

    def read(self):
        return json.dumps({"message": {"content": self._content}}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_client_chat_retries_then_succeeds(monkeypatch):
    import keith_llm.llm as llm_mod

    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("transient")
        return _FakeResp("hello")

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", fake_urlopen)
    assert OllamaClient("m", retries=2).chat("hi") == "hello"
    assert calls["n"] == 2  # failed once, succeeded on retry


def test_client_chat_raises_after_exhausting_retries(monkeypatch):
    import pytest

    import keith_llm.llm as llm_mod

    def always_fail(req, timeout=None):
        raise OSError("down")

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", always_fail)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        OllamaClient("m", retries=2).chat("hi")


# --- build integration with a mocked generator ---


def _fixture_mirror(tmp_path):
    data = tmp_path / "data" / "bestiary"
    data.mkdir(parents=True)
    (data / "index.json").write_text(json.dumps({"MM": "bestiary-mm.json"}))
    (data / "bestiary-mm.json").write_text(
        json.dumps({"monster": [MON, {"name": "Goblin", "ac": [12]}]})
    )
    return tmp_path.as_uri()


def test_build_with_ollama_generator(tmp_path, monkeypatch):
    from keith_llm.sft import build as build_mod

    # make the client "available" and produce fixed synthetic pairs
    monkeypatch.setattr(build_mod.OllamaClient, "available", lambda self: True)
    monkeypatch.setattr(
        build_mod, "synth_monster_pairs", lambda client, mon, n=5: [("synth q?", "synth a.")]
    )
    out = tmp_path / "sft.jsonl"
    stats = build_sft_dataset(out, base_url=_fixture_mirror(tmp_path), generator="ollama")
    assert stats["generator"] == "ollama"
    assert stats["bestiary"] > 0
    text = out.read_text()
    assert "synth q?" in text  # LLM-generated pairs are in the dataset


def test_build_ollama_unavailable_errors(tmp_path, monkeypatch):
    import pytest

    from keith_llm.sft import build as build_mod

    monkeypatch.setattr(build_mod.OllamaClient, "available", lambda self: False)
    with pytest.raises(SystemExit, match="ollama not reachable"):
        build_sft_dataset(
            tmp_path / "x.jsonl", base_url=_fixture_mirror(tmp_path), generator="ollama"
        )


def test_build_both_combines_programmatic_and_synth(tmp_path, monkeypatch):
    from keith_llm.sft import build as build_mod

    monkeypatch.setattr(build_mod.OllamaClient, "available", lambda self: True)
    monkeypatch.setattr(
        build_mod, "synth_monster_pairs", lambda client, mon, n=5: [("synth q?", "synth a.")]
    )
    out = tmp_path / "sft.jsonl"
    build_sft_dataset(out, base_url=_fixture_mirror(tmp_path), generator="both")
    text = out.read_text()
    assert "synth q?" in text  # synth pairs present
    assert "Armor Class" in text  # programmatic pairs also present


# --- corpus-grounded multi-system generation ---

DOC = {
    "system": "shadowrun",
    "doc_type": "rules",
    "source": "shadowrun/core.txt",
    "text": "The Johnson pays runners in nuyen. " * 40,
}


def test_synth_corpus_pairs_grounds_on_text_and_system():
    client = _FakeClient('[{"instruction": "Who pays runners?", "response": "The Johnson."}]')
    pairs = synth_corpus_pairs(client, DOC, n_pairs=3)
    assert pairs == [("Who pays runners?", "The Johnson.")]
    # the real text + system are in the prompt -> grounded, not invented
    assert "nuyen" in client.last_prompt
    assert "shadowrun" in client.last_prompt


def test_synth_corpus_pairs_skips_short_docs():
    assert synth_corpus_pairs(_FakeClient("[]"), {"system": "d6", "text": "too short"}) == []


def test_synth_corpus_pairs_failure_returns_empty():
    assert synth_corpus_pairs(_FakeClient("", boom=True), DOC) == []


def _write_corpus(tmp_path, records):
    p = tmp_path / "corpus.jsonl"
    with p.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return p


def test_sample_corpus_docs_is_balanced_per_system(tmp_path):
    import random

    recs = (
        [{"system": "dnd5e", "text": f"a{i}"} for i in range(5)]
        + [{"system": "shadowrun", "text": f"b{i}"} for i in range(3)]
        + [{"system": "d6", "text": "c0"}]
    )
    corpus = _write_corpus(tmp_path, recs)
    docs = _sample_corpus_docs(corpus, docs_per_system=2, rng=random.Random(0))
    from collections import Counter

    per = Counter(d["system"] for d in docs)
    assert per == {"dnd5e": 2, "shadowrun": 2, "d6": 1}  # capped at 2, small systems intact


def test_sample_corpus_docs_skips_non_object_lines(tmp_path):
    import random

    # valid-JSON but non-object lines (null, array, number) must not crash sampling
    p = tmp_path / "corpus.jsonl"
    p.write_text('null\n[1,2,3]\n42\n{"system":"d6","text":"ok"}\n')
    docs = _sample_corpus_docs(p, docs_per_system=5, rng=random.Random(0))
    assert [d["system"] for d in docs] == ["d6"]


def test_build_from_corpus_adds_multisystem_pairs(tmp_path, monkeypatch):
    from keith_llm.sft import build as build_mod

    monkeypatch.setattr(build_mod.OllamaClient, "available", lambda self: True)
    monkeypatch.setattr(
        build_mod,
        "synth_corpus_pairs",
        lambda client, doc, n=5: [(f"about {doc['system']}?", "grounded answer.")],
    )
    corpus = _write_corpus(
        tmp_path,
        [
            {"system": "shadowrun", "text": "x " * 100},
            {"system": "call_of_cthulhu", "text": "y " * 100},
        ],
    )
    out = tmp_path / "sft.jsonl"
    stats = build_sft_dataset(out, from_corpus=corpus, corpus_docs_per_system=5)
    assert stats["corpus"] == 2
    text = out.read_text()
    assert "about shadowrun?" in text
    assert "about call_of_cthulhu?" in text
    assert '"source": "corpus/shadowrun"' in text


def test_build_from_corpus_errors_when_ollama_down(tmp_path, monkeypatch):
    import pytest

    from keith_llm.sft import build as build_mod

    monkeypatch.setattr(build_mod.OllamaClient, "available", lambda self: False)
    corpus = _write_corpus(tmp_path, [{"system": "d6", "text": "z " * 100}])
    with pytest.raises(SystemExit, match="ollama not reachable"):
        build_sft_dataset(tmp_path / "x.jsonl", from_corpus=corpus)
