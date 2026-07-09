"""Import a self-hosted 5etools mirror as raw training text.

5etools stores content as recursive "entries" JSON with inline tags like
``{@creature mummy lord|MM|the mummy}``. This module renders that to plain
text and writes one .txt per adventure/book/bestiary-source into a
``data/raw/dnd5e/{adventure,rules,bestiary}/`` layout that ``keith-llm
ingest`` already understands.

The fetched content is proprietary WotC material: it belongs in the
gitignored data/raw/ tree and must stay ``publishable: false`` in the
manifest. Only this tooling lives in the repo.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TAG_WITH_BODY = re.compile(r"\{@(\w+) ([^{}]*)\}")
_TAG_BARE = re.compile(r"\{@(\w+)\}")

_ATK_CODES = {
    "mw": "Melee Weapon Attack:",
    "rw": "Ranged Weapon Attack:",
    "mw,rw": "Melee or Ranged Weapon Attack:",
    "ms": "Melee Spell Attack:",
    "rs": "Ranged Spell Attack:",
    "ms,rs": "Melee or Ranged Spell Attack:",
}
_BARE_TAGS = {"h": "Hit: ", "atk": "Attack:", "recharge": "(Recharge 6)"}
# Tags whose first pipe segment is already display text.
_LITERAL_TAGS = {"b", "bold", "i", "italic", "s", "u", "note", "code", "dice", "damage", "d20"}


def _untag(text: str) -> str:
    """Strip 5etools inline tags, innermost first, until stable."""

    def repl(match: re.Match[str]) -> str:
        tag, body = match.group(1), match.group(2)
        parts = body.split("|")
        if tag == "atk":
            return _ATK_CODES.get(parts[0], "Attack:")
        if tag == "dc":
            return f"DC {parts[0]}"
        if tag == "hit":
            val = parts[0]
            return val if val.startswith(("+", "-")) else f"+{val}"
        if tag == "recharge":
            return f"(Recharge {parts[0]}-6)" if parts[0] else "(Recharge 6)"
        if tag in _LITERAL_TAGS:
            return parts[0]
        # Reference tags: {@creature name|source|display} -> display if given.
        return parts[2] if len(parts) >= 3 and parts[2] else parts[0]

    while True:
        new = _TAG_WITH_BODY.sub(repl, text)
        new = _TAG_BARE.sub(lambda m: _BARE_TAGS.get(m.group(1), ""), new)
        if new == text:
            return new
        text = new


def _render_table(node: dict[str, Any], out: list[str]) -> None:
    if caption := node.get("caption"):
        out.append(str(caption))
    labels = node.get("colLabels")
    if labels:
        out.append(" | ".join(_untag(str(c)) for c in labels))
    for row in node.get("rows", []):
        cells = row if isinstance(row, list) else [row]
        rendered = []
        for cell in cells:
            if isinstance(cell, dict):
                roll = cell.get("roll", {})
                text = roll.get("exact", f"{roll.get('min', '')}-{roll.get('max', '')}")
                rendered.append(str(text))
            else:
                rendered.append(_untag(str(cell)))
        out.append(" | ".join(rendered))


def render_entries(node: Any, out: list[str] | None = None) -> str:
    """Walk any 5etools entries tree and accumulate plain-text lines."""
    top = out is None
    if out is None:
        out = []
    if isinstance(node, str):
        out.append(_untag(node))
    elif isinstance(node, int | float):
        out.append(str(node))
    elif isinstance(node, list):
        for child in node:
            render_entries(child, out)
    elif isinstance(node, dict):
        ntype = node.get("type", "entries")
        if ntype == "image":
            return "" if top else ""
        if name := node.get("name"):
            out.append(f"\n{_untag(str(name))}\n")
        if ntype == "table":
            _render_table(node, out)
        elif ntype == "list":
            for item in node.get("items", []):
                sub: list[str] = []
                render_entries(item, sub)
                joined = " ".join(s.strip() for s in sub if s.strip())
                if joined:
                    out.append(f"- {joined}")
        else:
            for key in ("headerEntries", "entries", "entry", "footerEntries", "rows"):
                if key in node:
                    render_entries(node[key], out)
        if ntype == "quote" and (by := node.get("by")):
            out.append(f"— {_untag(str(by))}")
    return "\n".join(out).strip() if top else ""


def _fmt_speed(speed: Any) -> str:
    if isinstance(speed, dict):
        parts = []
        for mode, val in speed.items():
            if mode == "canHover":
                continue
            dist = val.get("number", "") if isinstance(val, dict) else val
            parts.append(f"{mode} {dist} ft.")
        return ", ".join(parts)
    return str(speed)


def render_monster(mon: dict[str, Any]) -> str:
    """Lenient plain-text stat block."""
    lines = [f"\n{mon.get('name', 'Unknown Creature')}\n"]
    mtype = mon.get("type")
    if isinstance(mtype, dict):
        mtype = mtype.get("type", "")
    size = mon.get("size")
    if isinstance(size, list):
        size = size[0] if size else ""
    meta = " ".join(str(x) for x in (size or "", mtype) if x)
    if align := mon.get("alignment"):
        if isinstance(align, list):
            meta += ", " + " ".join(str(a) for a in align if isinstance(a, str))
    lines.append(meta)

    if ac := mon.get("ac"):
        first = ac[0] if isinstance(ac, list) else ac
        lines.append(f"Armor Class {first.get('ac') if isinstance(first, dict) else first}")
    if hp := mon.get("hp"):
        if isinstance(hp, dict):
            lines.append(f"Hit Points {hp.get('average', '')} ({hp.get('formula', '')})")
    if speed := mon.get("speed"):
        lines.append(f"Speed {_fmt_speed(speed)}")
    stats = [mon.get(k) for k in ("str", "dex", "con", "int", "wis", "cha")]
    if all(s is not None for s in stats):
        lines.append("STR {} DEX {} CON {} INT {} WIS {} CHA {}".format(*stats))
    for label, key in (("Skills", "skill"), ("Senses", "senses"), ("Languages", "languages")):
        if val := mon.get(key):
            if isinstance(val, dict):
                val = ", ".join(f"{k} {v}" for k, v in val.items())
            elif isinstance(val, list):
                val = ", ".join(str(v) for v in val)
            lines.append(f"{label} {val}")
    if cr := mon.get("cr"):
        lines.append(f"Challenge {cr.get('cr') if isinstance(cr, dict) else cr}")

    for group in ("trait", "action", "bonus", "reaction", "legendary", "mythic"):
        for block in mon.get(group) or []:
            sub: list[str] = []
            render_entries(block, sub)
            text = "\n".join(s for s in sub if s.strip())
            if text:
                lines.append(text)
    return "\n".join(lines).strip()


def _get_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - user-supplied mirror
        return json.load(resp)


def _write(path: Path, title: str, body: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"{title}\n\n{body}\n"
    path.write_text(text)
    return len(text)


def fetch_all(
    base_url: str,
    out_dir: str | Path,
    categories: tuple[str, ...] = ("adventures", "books", "bestiary"),
) -> dict[str, Any]:
    """Download and convert a 5etools mirror. Returns per-category stats."""
    base = base_url.rstrip("/")
    out_dir = Path(out_dir)
    stats = {"files": 0, "characters": 0, "failed": []}

    def grab(kind: str, index_key: str, subdir: str) -> None:
        index = _get_json(f"{base}/data/{index_key}.json")[index_key.rstrip("s") or index_key]
        for entry in index:
            ident = entry["id"]
            try:
                doc = _get_json(f"{base}/data/{kind}/{kind}-{ident.lower()}.json")
                body = render_entries(doc.get("data", []))
                n = _write(out_dir / subdir / f"{ident}.txt", entry.get("name", ident), body)
                stats["files"] += 1
                stats["characters"] += n
            except Exception as exc:  # noqa: BLE001 - keep going past one bad file
                logger.warning("failed %s %s: %s", kind, ident, exc)
                stats["failed"].append(f"{kind}/{ident}")

    if "adventures" in categories:
        grab("adventure", "adventures", "adventure")
    if "books" in categories:
        grab("book", "books", "rules")

    if "bestiary" in categories:
        fluff_index: dict[str, str] = {}
        try:
            fluff_index = _get_json(f"{base}/data/bestiary/fluff-index.json")
        except Exception:  # noqa: BLE001
            logger.warning("no bestiary fluff index; skipping lore text")
        index = _get_json(f"{base}/data/bestiary/index.json")
        for source, filename in index.items():
            try:
                monsters = _get_json(f"{base}/data/bestiary/{filename}").get("monster", [])
                parts = [render_monster(m) for m in monsters if not m.get("_copy")]
                if source in fluff_index:
                    fluff = _get_json(f"{base}/data/bestiary/{fluff_index[source]}")
                    for entry in fluff.get("monsterFluff", []):
                        sub: list[str] = []
                        render_entries(entry.get("entries", []), sub)
                        text = "\n".join(s for s in sub if s.strip())
                        if text:
                            parts.append(f"\n{entry.get('name', '')}\n{text}")
                body = "\n\n".join(p for p in parts if p)
                if not body:
                    continue
                n = _write(out_dir / "bestiary" / f"{source}.txt", f"Creatures of {source}", body)
                stats["files"] += 1
                stats["characters"] += n
            except Exception as exc:  # noqa: BLE001
                logger.warning("failed bestiary %s: %s", source, exc)
                stats["failed"].append(f"bestiary/{source}")

    logger.info("5etools import -> %s: %s", out_dir, stats)
    return stats
