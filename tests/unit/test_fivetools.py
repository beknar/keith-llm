import json

import pytest

from keith_llm.data.fivetools import _untag, fetch_all, render_entries, render_monster

# --- tag stripping ---


@pytest.mark.parametrize(
    ("tagged", "expected"),
    [
        ("roll {@dice 2d6+3} damage", "roll 2d6+3 damage"),
        ("the {@creature mummy lord|MM|the mummy} attacks", "the the mummy attacks"),
        ("cast {@spell fireball}", "cast fireball"),
        ("{@creature goblin|MM}", "goblin"),
        ("{@b {@dice 1d4}} bolts", "1d4 bolts"),
        ("{@atk mw} {@hit 7} to hit", "Melee Weapon Attack: +7 to hit"),
        ("{@h}12 ({@damage 2d8+3}) piercing", "Hit: 12 (2d8+3) piercing"),
        ("save {@dc 15} or else", "save DC 15 or else"),
        ("{@recharge 5} breath", "(Recharge 5-6) breath"),
        ("plain text stays", "plain text stays"),
    ],
)
def test_untag(tagged, expected):
    assert _untag(tagged) == expected


# --- entries rendering ---


def test_render_nested_sections_lists_tables():
    doc = [
        {
            "type": "section",
            "name": "Chapter 1: Emberfall",
            "entries": [
                "The village sits at the edge of the {@i mirewood}.",
                {
                    "type": "list",
                    "items": ["Torchlight flickers", {"type": "entries", "entries": ["A barrow"]}],
                },
                {
                    "type": "table",
                    "caption": "Random Encounters",
                    "colLabels": ["d6", "Encounter"],
                    "rows": [
                        ["1", "{@creature goblin|MM} ambush"],
                        [{"roll": {"min": 2, "max": 6}}, "Nothing"],
                    ],
                },
            ],
        }
    ]
    text = render_entries(doc)
    assert "Chapter 1: Emberfall" in text
    assert "edge of the mirewood" in text
    assert "- Torchlight flickers" in text
    assert "- A barrow" in text
    assert "d6 | Encounter" in text
    assert "1 | goblin ambush" in text
    assert "2-6 | Nothing" in text


def test_render_skips_images():
    assert render_entries([{"type": "image", "href": {"path": "x.png"}}]) == ""


def test_render_quote_attribution():
    text = render_entries([{"type": "quote", "entries": ["Beware the mire."], "by": "Elder Ronn"}])
    assert "Beware the mire." in text
    assert "— Elder Ronn" in text


# --- monster rendering ---


def test_render_monster_statblock():
    mon = {
        "name": "Mire Goblin",
        "size": ["S"],
        "type": {"type": "humanoid"},
        "alignment": ["N", "E"],
        "ac": [{"ac": 15, "from": ["leather armor"]}],
        "hp": {"average": 7, "formula": "2d6"},
        "speed": {"walk": 30, "swim": 20},
        "str": 8,
        "dex": 14,
        "con": 10,
        "int": 10,
        "wis": 8,
        "cha": 8,
        "senses": ["darkvision 60 ft."],
        "cr": "1/4",
        "trait": [{"name": "Bog Camouflage", "entries": ["Advantage on Stealth in swamps."]}],
        "action": [
            {"name": "Scimitar", "entries": ["{@atk mw} {@hit 4} to hit. {@h}5 ({@damage 1d6+2})."]}
        ],
    }
    text = render_monster(mon)
    assert "Mire Goblin" in text
    assert "Armor Class 15" in text
    assert "Hit Points 7 (2d6)" in text
    assert "walk 30 ft., swim 20 ft." in text
    assert "STR 8 DEX 14 CON 10 INT 10 WIS 8 CHA 8" in text
    assert "Challenge 1/4" in text
    assert "Bog Camouflage" in text
    assert "Melee Weapon Attack: +4 to hit. Hit: 5 (1d6+2)." in text


# --- fetch_all against a file:// fixture mirror ---


@pytest.fixture()
def mirror(tmp_path):
    data = tmp_path / "data"
    (data / "adventure").mkdir(parents=True)
    (data / "book").mkdir()
    (data / "bestiary").mkdir()
    (data / "adventures.json").write_text(
        json.dumps({"adventure": [{"id": "LMoP", "name": "Lost Mine of Phandelver"}]})
    )
    (data / "adventure" / "adventure-lmop.json").write_text(
        json.dumps(
            {"data": [{"type": "section", "name": "Goblin Arrows", "entries": ["An ambush."]}]}
        )
    )
    (data / "books.json").write_text(
        json.dumps({"book": [{"id": "PHB", "name": "Player's Handbook"}]})
    )
    (data / "book" / "book-phb.json").write_text(
        json.dumps({"data": [{"type": "section", "name": "Races", "entries": ["Choose a race."]}]})
    )
    (data / "bestiary" / "index.json").write_text(json.dumps({"MM": "bestiary-mm.json"}))
    (data / "bestiary" / "bestiary-mm.json").write_text(
        json.dumps({"monster": [{"name": "Goblin", "hp": {"average": 7, "formula": "2d6"}}]})
    )
    (data / "bestiary" / "fluff-index.json").write_text(
        json.dumps({"MM": "fluff-bestiary-mm.json"})
    )
    (data / "bestiary" / "fluff-bestiary-mm.json").write_text(
        json.dumps(
            {"monsterFluff": [{"name": "Goblin", "entries": ["Goblins are small menaces."]}]}
        )
    )
    return tmp_path.as_uri()


def test_fetch_all(mirror, tmp_path):
    out = tmp_path / "out"
    stats = fetch_all(mirror, out)
    assert stats["files"] == 3
    assert stats["failed"] == []
    adventure = (out / "adventure" / "LMoP.txt").read_text()
    assert adventure.startswith("Lost Mine of Phandelver")
    assert "Goblin Arrows" in adventure and "An ambush." in adventure
    rules = (out / "rules" / "PHB.txt").read_text()
    assert rules.startswith("Player's Handbook") and "Choose a race." in rules
    bestiary = (out / "bestiary" / "MM.txt").read_text()
    assert "Goblin" in bestiary and "Goblins are small menaces." in bestiary


def test_fetch_all_survives_missing_file(mirror, tmp_path):
    # Break one adventure file; the rest still imports.
    import urllib.parse

    root = urllib.parse.urlparse(mirror).path
    (tmp_path / "out").mkdir()
    import pathlib

    pathlib.Path(root, "data", "adventure", "adventure-lmop.json").unlink()
    stats = fetch_all(mirror, tmp_path / "out")
    assert "adventure/LMoP" in stats["failed"]
    assert stats["files"] == 2
