from pathlib import Path

from keith_llm.data.classify import (
    apply_moves,
    classify_file,
    classify_paths,
    classify_text,
)

BESTIARY = (
    "Mummy Lord\nMedium undead, lawful evil\nArmor Class 17\nHit Points 97\n"
    "Speed 20 ft.\nSTR 18 DEX 10 CON 17 INT 11 WIS 18 CHA 16\n"
    "Damage Immunities necrotic, poison\nSenses darkvision 60 ft.\nChallenge 15\n"
    "Multiattack. The mummy can use its Dreadful Glare and makes one attack with its fist.\n"
    "Rotting Fist. Melee Weapon Attack: +9 to hit. Legendary Actions. The mummy lord can take "
    "3 legendary actions. Recharge 5-6. Damage resistance to bludgeoning. "
) * 6

ADVENTURE = (
    "Chapter 1: Into the Mire\nWhen the characters arrive at the village, read aloud the boxed "
    "text below. The party sees a ruined watchtower. Area 3: The Crypt. If the characters open "
    "the sarcophagus, a guardian awakens. Development: once the party defeats it, the dungeon "
    "master should describe the treasure. The adventurers can rest here. "
) * 8

RULES = (
    "Chapter 2: Playing the Game\nWhen you make an ability check, roll a d20 and add your "
    "proficiency bonus. At higher levels you gain additional features. A saving throw resists "
    "an effect. The rules for combat follow. When you reach 3rd-level you gain a subclass. "
    "You gain hit points each level. "
) * 8

MIXED_LOW = "A short note about the weather and some travel between towns and a market. " * 3


def test_classify_bestiary():
    dt, conf, _ = classify_text(BESTIARY)
    assert dt == "bestiary"
    assert conf > 0.45


def test_classify_adventure():
    dt, conf, _ = classify_text(ADVENTURE)
    assert dt == "adventure"
    assert conf > 0.45


def test_classify_rules():
    dt, conf, _ = classify_text(RULES)
    assert dt == "rules"
    assert conf > 0.45


def test_no_signal_is_none_and_zero_confidence():
    dt, conf, scores = classify_text("just some plain words with no gaming markers at all here")
    assert dt is None
    assert conf == 0.0


def test_two_way_tie_is_low_confidence():
    # Equal bestiary and adventure signal must NOT be confidently routed:
    # confidence is the winner's lead over the runner-up, so a tie scores ~0.
    # bestiary: armor class(3)+challenge(2)=5; adventure: the party(3)+treasure(1)+development(1)=5
    text = "Armor Class 15. Challenge 5. The party finds treasure. Development follows."
    _, conf, scores = classify_text(text)
    top2 = sorted(scores.values(), reverse=True)[:2]
    assert abs(top2[0] - top2[1]) < 1e-9  # genuinely tied
    assert conf < 0.45  # -> flagged REVIEW, not moved


def test_classify_paths_survives_unreadable_file(tmp_path, monkeypatch):
    from keith_llm.data import classify as classify_mod

    src = tmp_path / "unsorted"
    src.mkdir()
    (src / "good.txt").write_text(BESTIARY)
    (src / "bad.txt").write_text("whatever")

    real = classify_mod.classify_file

    def flaky(path, *a, **k):
        if path.name == "bad.txt":
            raise OSError("simulated unreadable file")
        return real(path, *a, **k)

    monkeypatch.setattr(classify_mod, "classify_file", flaky)
    results = classify_paths(src, system="dnd5e", dest=tmp_path / "raw")
    # the bad file is skipped, the good one still classified
    assert [Path(r.path).name for r in results] == ["good.txt"]


def test_classify_file_txt(tmp_path):
    f = tmp_path / "monsters.txt"
    f.write_text(BESTIARY)
    r = classify_file(f, system="dnd5e", dest=tmp_path / "raw")
    assert r.doc_type == "bestiary"
    assert r.confident
    assert r.target == str(tmp_path / "raw" / "dnd5e" / "bestiary" / "monsters.txt")


def test_classify_file_low_confidence_not_moved(tmp_path):
    f = tmp_path / "vague.txt"
    f.write_text(MIXED_LOW)
    r = classify_file(f, system="dnd5e", dest=tmp_path / "raw", min_confidence=0.9)
    assert not r.confident
    assert r.target is None


def test_classify_file_skips_unsupported(tmp_path):
    f = tmp_path / "art.png"
    f.write_bytes(b"\x89PNG")
    assert classify_file(f, system="dnd5e", dest=tmp_path) is None


def test_classify_paths_directory(tmp_path):
    src = tmp_path / "unsorted"
    src.mkdir()
    (src / "mm.txt").write_text(BESTIARY)
    (src / "lmop.txt").write_text(ADVENTURE)
    (src / "phb.txt").write_text(RULES)
    (src / "cover.jpg").write_bytes(b"x")  # skipped
    results = classify_paths(src, system="dnd5e", dest=tmp_path / "raw")
    by_name = {Path(r.path).name: r.doc_type for r in results}
    assert by_name == {"mm.txt": "bestiary", "lmop.txt": "adventure", "phb.txt": "rules"}


def test_apply_moves_relocates_and_preserves(tmp_path):
    src = tmp_path / "unsorted"
    src.mkdir()
    (src / "mm.txt").write_text(BESTIARY)
    results = classify_paths(src, system="dnd5e", dest=tmp_path / "raw")
    out = apply_moves(results)
    assert len(out["moved"]) == 1
    moved_to = tmp_path / "raw" / "dnd5e" / "bestiary" / "mm.txt"
    assert moved_to.exists()
    assert not (src / "mm.txt").exists()  # relocated, not copied
    assert moved_to.read_text() == BESTIARY


def test_apply_moves_skips_existing_target(tmp_path):
    src = tmp_path / "unsorted"
    src.mkdir()
    (src / "mm.txt").write_text(BESTIARY)
    dest = tmp_path / "raw"
    existing = dest / "dnd5e" / "bestiary" / "mm.txt"
    existing.parent.mkdir(parents=True)
    existing.write_text("already here")
    results = classify_paths(src, system="dnd5e", dest=dest)
    out = apply_moves(results)
    assert out["moved"] == []
    assert len(out["skipped"]) == 1
    assert (src / "mm.txt").exists()  # left in place
    assert existing.read_text() == "already here"  # not overwritten
