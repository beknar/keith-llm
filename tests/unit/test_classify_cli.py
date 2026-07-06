import builtins

from keith_llm import cli

BESTIARY = (
    "Mummy Lord\nMedium undead, lawful evil\nArmor Class 17\nHit Points 97\n"
    "Speed 20 ft.\nSTR 18 DEX 10 CON 17 INT 11 WIS 18 CHA 16\n"
    "Damage Immunities necrotic, poison\nSenses darkvision 60 ft.\nChallenge 15\n"
    "Multiattack. The mummy makes one attack with its fist.\n"
    "Rotting Fist. Melee Weapon Attack: +9 to hit. Legendary Actions. The mummy lord can take "
    "3 legendary actions. Recharge 5-6. Damage resistance to bludgeoning. "
) * 6


def _setup(tmp_path):
    src = tmp_path / "unsorted"
    src.mkdir()
    (src / "mm.txt").write_text(BESTIARY)
    return src


def test_cli_prompt_yes_moves(tmp_path, monkeypatch):
    src = _setup(tmp_path)
    monkeypatch.setattr(builtins, "input", lambda *a: "y")
    rc = cli.main(["classify", "--src", str(src), "--dest", str(tmp_path / "raw")])
    assert rc == 0
    assert (tmp_path / "raw" / "dnd5e" / "bestiary" / "mm.txt").exists()
    assert not (src / "mm.txt").exists()


def test_cli_prompt_no_keeps(tmp_path, monkeypatch):
    src = _setup(tmp_path)
    monkeypatch.setattr(builtins, "input", lambda *a: "n")
    rc = cli.main(["classify", "--src", str(src), "--dest", str(tmp_path / "raw")])
    assert rc == 0
    assert (src / "mm.txt").exists()  # untouched
    assert not (tmp_path / "raw" / "dnd5e" / "bestiary" / "mm.txt").exists()


def test_cli_dry_run_never_moves(tmp_path, monkeypatch):
    src = _setup(tmp_path)

    def no_prompt(*a):
        raise AssertionError("dry-run must not prompt")

    monkeypatch.setattr(builtins, "input", no_prompt)
    rc = cli.main(["classify", "--src", str(src), "--dest", str(tmp_path / "raw"), "--dry-run"])
    assert rc == 0
    assert (src / "mm.txt").exists()


def test_cli_yes_flag_skips_prompt(tmp_path, monkeypatch):
    src = _setup(tmp_path)

    def no_prompt(*a):
        raise AssertionError("--yes must not prompt")

    monkeypatch.setattr(builtins, "input", no_prompt)
    rc = cli.main(["classify", "--src", str(src), "--dest", str(tmp_path / "raw"), "--yes"])
    assert rc == 0
    assert (tmp_path / "raw" / "dnd5e" / "bestiary" / "mm.txt").exists()
