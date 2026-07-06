"""Heuristic doc-type routing aid: suggest rules / bestiary / adventure.

``doc_type`` is otherwise assigned structurally (which folder a file sits in;
see data/sources.yaml). This module *suggests* a type from content so a pile of
unsorted files can be routed into the right ``data/raw/<system>/<doc_type>/``
directory instead of hand-sorted.

It is deliberately a suggestion, not an authority: real books are mixed (a
rulebook has a bestiary chapter; an adventure is full of stat blocks), so the
score is weighted keyword density and low-confidence files are flagged for
manual review rather than moved. Classification runs on a sample (the first few
pages) so scanned books aren't fully OCR'd just to be sorted.
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from keith_llm.constants import DOC_TYPES
from keith_llm.data.clean import clean_pages, clean_text
from keith_llm.data.ingest import TEXT_EXTS
from keith_llm.data.pdf_layout import extract_pdf_pages

logger = logging.getLogger(__name__)

_SAMPLE_PAGES = 12
_SAMPLE_CHARS = 60_000

# (pattern, weight) per doc type. Distinctive markers weigh more; the overlap
# between types is real, which is why weighting + density (not presence) drives
# the decision.
_RAW_SIGNALS: dict[str, list[tuple[str, float]]] = {
    "bestiary": [
        (r"\barmor class\b", 3),
        (r"\bhit points\b", 3),
        (r"\bchallenge\b", 2),
        (r"\blegendary action", 4),
        (r"\bmultiattack\b", 4),
        (r"\bmelee weapon attack\b", 3),
        (r"\bdamage (immunit|resistance|vulnerabilit)", 3),
        (r"\brecharge\b", 2),
        (r"\bstr\b.{0,15}\bdex\b.{0,15}\bcon\b", 4),
    ],
    "adventure": [
        (r"\bthe (characters|party|adventurers|heroes)\b", 3),
        (r"\bread[- ]?aloud\b", 4),
        (r"\bboxed text\b", 4),
        (r"\bif the (characters|party|players|adventurers)\b", 3),
        (r"\bwhen the (characters|party|adventurers) (enter|arrive|reach|open)", 3),
        (r"\b(area|room) \d+\b", 2),
        (r"\bdevelopment\b", 1),
        (r"\btreasure\b", 1),
        (r"\bdungeon master\b", 1),
    ],
    "rules": [
        (r"\bat higher levels\b", 4),
        (r"\bproficiency bonus\b", 3),
        (r"\bwhen you\b", 2),
        (r"\byou gain\b", 2),
        (r"\bability (check|score)\b", 2),
        (r"\bsaving throw\b", 1),
        (r"\b\d+(st|nd|rd|th)[- ]level\b", 2),
        (r"\bthe rules\b", 2),
        (r"\bchapter \d+\b", 2),
    ],
}
_SIGNALS = {
    dt: [(re.compile(p, re.IGNORECASE | re.DOTALL), w) for p, w in sigs]
    for dt, sigs in _RAW_SIGNALS.items()
}


@dataclass
class ClassifyResult:
    path: str
    doc_type: str | None  # None when no signals matched at all
    confidence: float
    confident: bool
    target: str | None  # where it would move, if confident
    scores: dict[str, float] = field(default_factory=dict)


def classify_text(text: str) -> tuple[str | None, float, dict[str, float]]:
    """Return (best_doc_type, confidence in [0,1], per-type scores).

    Confidence is the winner's LEAD over the runner-up as a share of total
    signal: a dominant type approaches 1.0, a 2-way tie is 0.0, and no signal
    at all returns (None, 0.0). Measuring the lead (not the winning share) keeps
    genuinely ambiguous files below threshold so they aren't confidently moved.
    """
    n_words = max(len(text.split()), 1)
    scale = 1000.0 / n_words  # signal rate per 1000 words, so length doesn't bias
    scores = {
        dt: sum(w * len(pat.findall(text)) for pat, w in sigs) * scale
        for dt, sigs in _SIGNALS.items()
    }
    total = sum(scores.values())
    if total <= 0:
        return None, 0.0, scores
    ordered = sorted(scores.values(), reverse=True)
    best = max(scores, key=scores.__getitem__)
    confidence = (ordered[0] - ordered[1]) / total
    return best, confidence, scores


def _sample_text(path: Path, enable_ocr: bool) -> str | None:
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTS:
        return path.read_text(encoding="utf-8", errors="replace")[:_SAMPLE_CHARS]
    if suffix == ".pdf":
        pages = extract_pdf_pages(str(path), enable_ocr=enable_ocr, max_pages=_SAMPLE_PAGES)
        return clean_text(clean_pages(pages))[:_SAMPLE_CHARS]
    return None  # unsupported / archive — not classified


def classify_file(
    path: Path,
    system: str,
    dest: Path,
    enable_ocr: bool = True,
    min_confidence: float = 0.45,
) -> ClassifyResult | None:
    text = _sample_text(path, enable_ocr)
    if text is None:
        return None
    doc_type, confidence, scores = classify_text(text)
    confident = doc_type is not None and confidence >= min_confidence
    target = str(dest / system / doc_type / path.name) if confident else None
    return ClassifyResult(str(path), doc_type, round(confidence, 3), confident, target, scores)


def classify_paths(
    src: str | Path,
    system: str,
    dest: str | Path = "data/raw",
    enable_ocr: bool = True,
    min_confidence: float = 0.45,
) -> list[ClassifyResult]:
    """Classify every supported document under ``src`` (a file or directory)."""
    src = Path(src)
    dest = Path(dest)
    paths = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.is_file())
    results = []
    for p in paths:
        try:
            result = classify_file(p, system, dest, enable_ocr, min_confidence)
        except Exception as exc:  # noqa: BLE001 - one unreadable file must not kill the batch
            logger.warning("could not classify %s: %s", p, exc)
            continue
        if result is not None:
            results.append(result)
    return results


def apply_moves(results: list[ClassifyResult]) -> dict[str, Any]:
    """Move each confident result to its target. Existing targets are skipped
    (never overwritten); the move preserves the file, only relocating it."""
    moved: list[str] = []
    skipped: list[str] = []
    for r in results:
        if not r.confident or r.target is None:
            continue
        target = Path(r.target)
        if target.exists():
            logger.warning("target exists, skipping: %s", target)
            skipped.append(r.path)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(r.path, str(target))
        moved.append(f"{r.path} -> {r.target}")
    logger.info("classify: moved %d files, skipped %d", len(moved), len(skipped))
    return {"moved": moved, "skipped": skipped}


def _validate_doc_types() -> None:
    unknown = set(_SIGNALS) - set(DOC_TYPES)
    if unknown:
        raise ValueError(f"classify signals reference unknown doc types: {sorted(unknown)}")


_validate_doc_types()
