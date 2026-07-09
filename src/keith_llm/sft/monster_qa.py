"""Grounded instruction/response pairs from 5etools monster stat blocks.

Field-level Q/A ("What is the Armor Class of X?") is generated straight from
the structured JSON, so the answers are correct by construction — the ideal
grounding for factual questions a small model would otherwise hallucinate.
Multiple phrasings per field add variety so the model doesn't just memorize one
template. The reference stat-block answer reuses ``fivetools.render_monster``.
"""

from __future__ import annotations

import random
from typing import Any

from keith_llm.data.fivetools import render_monster

Pair = tuple[str, str]

_ABILITIES = [
    ("str", "Strength"),
    ("dex", "Dexterity"),
    ("con", "Constitution"),
    ("int", "Intelligence"),
    ("wis", "Wisdom"),
    ("cha", "Charisma"),
]


def _first(value: Any) -> Any:
    return value[0] if isinstance(value, list) and value else value


def _type_str(mon: dict[str, Any]) -> str | None:
    mtype = mon.get("type")
    if isinstance(mtype, dict):
        mtype = mtype.get("type")
    size_map = {
        "T": "Tiny",
        "S": "Small",
        "M": "Medium",
        "L": "Large",
        "H": "Huge",
        "G": "Gargantuan",
    }
    size = size_map.get(str(_first(mon.get("size"))), "")
    if not mtype:
        return None
    parts = " ".join(p for p in (size, str(mtype)) if p)
    align = mon.get("alignment")
    if isinstance(align, list):
        align = " ".join(a for a in align if isinstance(a, str))
    return f"{parts}, {align}" if align else parts


def _ac(mon: dict[str, Any]) -> str | None:
    ac = _first(mon.get("ac"))
    if isinstance(ac, dict):
        ac = ac.get("ac")
    return str(ac) if ac is not None else None


def _hp(mon: dict[str, Any]) -> str | None:
    hp = mon.get("hp")
    if isinstance(hp, dict) and hp.get("average") is not None:
        formula = hp.get("formula")
        return f"{hp['average']} ({formula})" if formula else str(hp["average"])
    return None


def _speed(mon: dict[str, Any]) -> str | None:
    speed = mon.get("speed")
    if isinstance(speed, dict):
        parts = []
        for mode, val in speed.items():
            if mode == "canHover":
                continue
            dist = val.get("number") if isinstance(val, dict) else val
            parts.append(f"{mode} {dist} ft.")
        return ", ".join(parts) or None
    if speed is not None:
        return f"{speed} ft."
    return None


def _cr(mon: dict[str, Any]) -> str | None:
    cr = mon.get("cr")
    if isinstance(cr, dict):
        cr = cr.get("cr")
    return str(cr) if cr is not None else None


def _pick(rng: random.Random, templates: list[str], name: str) -> str:
    return rng.choice(templates).format(name=name)


def monster_qa(mon: dict[str, Any], rng: random.Random) -> list[Pair]:
    """Up to ~7 grounded pairs for one monster; skips fields it lacks."""
    name = mon.get("name")
    if not name or mon.get("_copy"):
        return []
    pairs: list[Pair] = []

    if (t := _type_str(mon)) is not None:
        pairs.append(
            (
                _pick(
                    rng,
                    [
                        "What kind of creature is the {name}?",
                        "What type of monster is a {name}?",
                    ],
                    name,
                ),
                f"The {name} is a {t}.",
            )
        )
    if (ac := _ac(mon)) is not None:
        pairs.append(
            (
                _pick(
                    rng,
                    [
                        "What is the Armor Class of the {name}?",
                        "What's the AC of a {name}?",
                        "How well armored is the {name}?",
                    ],
                    name,
                ),
                f"The {name} has an Armor Class of {ac}.",
            )
        )
    if (hp := _hp(mon)) is not None:
        pairs.append(
            (
                _pick(
                    rng,
                    [
                        "How many hit points does the {name} have?",
                        "What are the hit points of a {name}?",
                    ],
                    name,
                ),
                f"The {name} has {hp} hit points.",
            )
        )
    if (sp := _speed(mon)) is not None:
        pairs.append(
            (
                _pick(
                    rng,
                    [
                        "How fast can the {name} move?",
                        "What is the speed of the {name}?",
                    ],
                    name,
                ),
                f"The {name}'s speed is {sp}.",
            )
        )
    if (cr := _cr(mon)) is not None:
        pairs.append(
            (
                _pick(
                    rng,
                    [
                        "What is the challenge rating of the {name}?",
                        "What CR is a {name}?",
                    ],
                    name,
                ),
                f"The {name} has a challenge rating of {cr}.",
            )
        )
    abil = rng.choice(_ABILITIES)
    if mon.get(abil[0]) is not None:
        pairs.append(
            (f"What is the {name}'s {abil[1]} score?", f"The {name}'s {abil[1]} is {mon[abil[0]]}.")
        )
    block = render_monster(mon)
    if block.strip():
        pairs.append((f"Give me the stat block for the {name}.", block))
    return pairs
