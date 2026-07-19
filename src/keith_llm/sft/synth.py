"""Synthetic SFT pair generation via a local LLM (grounded, not invented).

The programmatic generator (monster_qa) is limited to a few fixed templates,
which trains a brittle lookup-table. This asks a local model to write *varied*
question/answer pairs — but **grounded**: it's given the real 5etools JSON and
told to use only those facts, so it supplies natural phrasing and question
variety without hallucinating stats. Generated pairs still pass through the
dataset's downstream filters.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from keith_llm.llm import OllamaClient

logger = logging.getLogger(__name__)

Pair = tuple[str, str]

_SYSTEM = (
    "You generate training examples for a tabletop-RPG assistant. Follow the "
    "instructions exactly and output only what is requested — no preamble, no "
    "commentary, no markdown fences."
)

_MONSTER_TMPL = (
    "Here is a Dungeons & Dragons monster as JSON:\n\n{data}\n\n"
    "Write {n} varied question-and-answer pairs a player or DM might ask about "
    "this creature. Rules:\n"
    "- Use ONLY facts present in the JSON. Do NOT invent anything.\n"
    "- Vary the questions (stats, abilities, actions, tactics, appearance, role).\n"
    "- Keep answers accurate and concise.\n"
    "Output ONLY a JSON array of objects like "
    '[{{"question": "...", "answer": "..."}}].'
)

_CORPUS_TMPL = (
    "Below is an excerpt from a tabletop RPG book (system: {system}):\n\n"
    "{text}\n\n"
    "Write {n} varied instruction/response training pairs grounded in this "
    "excerpt, for fine-tuning a tabletop-RPG assistant. Rules:\n"
    "- Use ONLY facts, names, and rules stated in the excerpt. Do NOT invent "
    "anything or rely on outside knowledge.\n"
    "- Vary the task types across the pairs: factual questions, 'Explain ...', "
    "'Describe ...', 'What happens when ...', summarize a rule or section, or a "
    "short creative prompt inspired by the content.\n"
    "- Make each instruction self-contained — the user will NOT see the excerpt.\n"
    "- Skip boilerplate (tables of contents, credits, page numbers).\n"
    "- Keep responses accurate and concise.\n"
    'Output ONLY a JSON array like [{{"instruction": "...", "response": "..."}}].'
)

_DECODER = json.JSONDecoder()


def _first_json_array(text: str) -> list[Any] | None:
    """Return the first well-formed top-level JSON array in ``text``, ignoring
    surrounding prose, code fences, or stray brackets. Scans each ``[`` and lets
    the JSON decoder find the matching close, so a bracket in the prose (or a
    second array) can't corrupt the span the way a first-to-last regex would."""
    start = text.find("[")
    while start != -1:
        try:
            value, _ = _DECODER.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find("[", start + 1)
            continue
        if isinstance(value, list):
            return value
        start = text.find("[", start + 1)
    return None


def parse_pairs(response: str) -> list[Pair]:
    """Extract (prompt, response) pairs from a model response. Accepts either
    ``{"question","answer"}`` or ``{"instruction","response"}`` keys. Robust to
    the model wrapping the JSON in prose or code fences; skips malformed items."""
    if not isinstance(response, str):
        return []
    items = _first_json_array(response)
    if items is None:
        return []
    pairs: list[Pair] = []
    for item in items:
        if isinstance(item, dict):
            q = str(item.get("instruction") or item.get("question") or "").strip()
            a = str(item.get("response") or item.get("answer") or "").strip()
            if q and a:
                pairs.append((q, a))
    return pairs


def synth_monster_pairs(client: OllamaClient, mon: dict[str, Any], n_pairs: int = 5) -> list[Pair]:
    """Generate grounded pairs for one monster. Returns [] on any failure so a
    single bad generation never aborts the build."""
    if mon.get("_copy") or not mon.get("name"):
        return []
    payload = json.dumps(
        {k: v for k, v in mon.items() if not k.startswith("_")}, ensure_ascii=False
    )
    try:
        response = client.chat(_MONSTER_TMPL.format(data=payload, n=n_pairs), system=_SYSTEM)
    except Exception as exc:  # noqa: BLE001 - one failed generation must not kill the build
        logger.warning("synth generation failed for %s: %s", mon.get("name"), exc)
        return []
    return parse_pairs(response)


def synth_corpus_pairs(
    client: OllamaClient, doc: dict[str, Any], n_pairs: int = 5, max_chars: int = 4000
) -> list[Pair]:
    """Generate varied, grounded instruction pairs from one corpus document's
    text (any system), so SFT data isn't limited to 5e monster Q/A. Uses a
    leading excerpt of the doc. Returns [] on any failure or if the doc is too
    short to ground anything useful."""
    text = (doc.get("text") or "").strip()
    if len(text) < 200:
        return []
    system = doc.get("system", "generic")
    try:
        response = client.chat(
            _CORPUS_TMPL.format(system=system, text=text[:max_chars], n=n_pairs),
            system=_SYSTEM,
        )
    except Exception as exc:  # noqa: BLE001 - one failed generation must not kill the build
        logger.warning("corpus synth failed for %s: %s", doc.get("source", "?"), exc)
        return []
    return parse_pairs(response)
