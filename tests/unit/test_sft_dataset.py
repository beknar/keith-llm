import json
import random
from pathlib import Path

from keith_llm.sft.build import _SEED_PATH, build_sft_dataset
from keith_llm.sft.format import (
    RESPONSE_HEADER,
    build_example_text,
    build_prompt,
)
from keith_llm.sft.monster_qa import monster_qa

MON = {
    "name": "Mire Goblin",
    "size": ["S"],
    "type": {"type": "humanoid"},
    "alignment": ["N", "E"],
    "ac": [{"ac": 15}],
    "hp": {"average": 7, "formula": "2d6"},
    "speed": {"walk": 30, "swim": 20},
    "str": 8,
    "dex": 14,
    "con": 10,
    "int": 10,
    "wis": 8,
    "cha": 8,
    "cr": "1/4",
    "action": [{"name": "Scimitar", "entries": ["{@atk mw} {@hit 4} to hit."]}],
}


# --- format ---


def test_build_prompt_and_example():
    p = build_prompt("  Do the thing  ")
    assert p.startswith("### Instruction:\nDo the thing")
    assert p.endswith(RESPONSE_HEADER)
    full = build_example_text("Q", "A")
    assert full == build_prompt("Q") + "A"


# --- monster Q/A grounding ---


def test_monster_qa_is_grounded():
    pairs = dict(monster_qa(MON, random.Random(0)))
    joined = " ".join(f"{q} :: {a}" for q, a in pairs.items())
    assert "Armor Class of 15" in joined
    assert "7 (2d6) hit points" in joined
    assert "challenge rating of 1/4" in joined
    assert "walk 30 ft., swim 20 ft." in joined
    # a "give me the stat block" pair whose answer is the rendered block
    block_q = [a for q, a in pairs.items() if "stat block" in q.lower()]
    assert block_q and "Mire Goblin" in block_q[0] and "Armor Class 15" in block_q[0]


def test_monster_qa_skips_copies_and_nameless():
    assert monster_qa({"name": "X", "_copy": {"name": "Y"}}, random.Random(0)) == []
    assert monster_qa({"ac": [12]}, random.Random(0)) == []


def test_monster_qa_handles_missing_fields():
    # only a name -> just the stat-block pair (or nothing), never crashes
    pairs = monster_qa({"name": "Blob"}, random.Random(0))
    assert all(isinstance(q, str) and isinstance(a, str) for q, a in pairs)


def test_monster_qa_deterministic():
    assert monster_qa(MON, random.Random(5)) == monster_qa(MON, random.Random(5))


# --- packaged seed ---


def test_seed_file_is_valid_jsonl():
    lines = [json.loads(ln) for ln in _SEED_PATH.read_text().splitlines() if ln.strip()]
    assert len(lines) >= 10
    assert all(set(rec) >= {"instruction", "response"} for rec in lines)
    assert all(rec["instruction"].strip() and rec["response"].strip() for rec in lines)


# --- build orchestration ---


def _fixture_mirror(tmp_path: Path) -> str:
    data = tmp_path / "data" / "bestiary"
    data.mkdir(parents=True)
    (data / "index.json").write_text(json.dumps({"MM": "bestiary-mm.json"}))
    (data / "bestiary-mm.json").write_text(
        json.dumps({"monster": [MON, {"name": "Goblin", "ac": [{"ac": 15}], "cr": "1/4"}]})
    )
    return tmp_path.as_uri()


def test_build_seed_only(tmp_path):
    out = tmp_path / "sft.jsonl"
    stats = build_sft_dataset(out, base_url=None)
    assert stats["grounded"] == 0
    assert stats["seed"] == stats["total"] >= 10
    records = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert all("instruction" in r and "response" in r and "source" in r for r in records)


def test_build_with_grounded(tmp_path):
    out = tmp_path / "sft.jsonl"
    stats = build_sft_dataset(out, base_url=_fixture_mirror(tmp_path))
    assert stats["grounded"] > 0
    assert stats["total"] == stats["seed"] + stats["grounded"]
    text = out.read_text()
    assert "Mire Goblin" in text and "Armor Class" in text


def test_build_deterministic(tmp_path):
    mirror = _fixture_mirror(tmp_path)
    a = build_sft_dataset(tmp_path / "a.jsonl", base_url=mirror, seed=3)
    b = build_sft_dataset(tmp_path / "b.jsonl", base_url=mirror, seed=3)
    assert a == b
    assert (tmp_path / "a.jsonl").read_text() == (tmp_path / "b.jsonl").read_text()


def test_build_missing_mirror_falls_back_to_seed(tmp_path):
    out = tmp_path / "sft.jsonl"
    stats = build_sft_dataset(out, base_url="file:///nonexistent/mirror")
    assert stats["grounded"] == 0
    assert stats["total"] == stats["seed"]  # seed still written, no crash
