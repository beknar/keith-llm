import json

from keith_llm.llm import OllamaClient
from keith_llm.sft.build import build_sft_dataset
from keith_llm.sft.synth import parse_pairs, synth_monster_pairs

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
    assert stats["grounded"] > 0
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
