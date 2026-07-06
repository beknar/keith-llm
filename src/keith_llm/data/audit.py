"""Corpus extraction-quality diagnostics.

Scores each document for signals that a PDF extracted badly, so you can spot
and fix problem sources *before* a long training run rather than after. The
existing :func:`keith_llm.data.clean.is_quality` is a pass/fail gate; this is
a graded report.

Signals (all dependency-free heuristics):

- ``alpha_ratio``       fraction of non-space characters that are letters.
- ``wordlike_frac``     of the alphabetic tokens, the fraction that look like
                        real words (contain a vowel, length 2-20). Low means
                        gibberish — often a broken text layer.
- ``internal_caps_rate``fraction of alphabetic tokens with an internal
                        lower->upper transition ("damageThe"). This is the
                        column-interleave / missing-space signature.
- ``long_token_frac``   fraction of whitespace tokens over 30 chars — runs of
                        concatenated words.
- ``garbage_line_frac`` fraction of non-blank lines that are mostly non-letters.
- ``words_per_line``    mean tokens per non-blank line. A text-dense document
                        reading ~1 word per line is the signature of pypdf's
                        column failure (words intact but one-per-line), which
                        the token metrics above miss.

Stat-block-heavy TTRPG text is legitimately low on ``alpha_ratio``, so the
BAD/WARN verdict leans on ``wordlike_frac`` and ``internal_caps_rate``, which
track genuine extraction failure rather than dense tables.
"""

from __future__ import annotations

import json
import logging
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_VOWEL = re.compile(r"[aeiouyAEIOUY]")
_INTERNAL_CAPS = re.compile(r"[a-z][A-Z]")
_STRIP = string.punctuation + string.digits + "’‘“”—–…•·"

VERDICT_ORDER = {"BAD": 0, "WARN": 1, "OK": 2}


def _alpha_cores(text: str) -> list[str]:
    """Whitespace tokens reduced to a leading/trailing-punctuation-stripped
    core, keeping only those that are wholly alphabetic and length >= 2 (so
    dice like ``2d6`` and stray single letters don't pollute the stats)."""
    cores = []
    for tok in text.split():
        core = tok.strip(_STRIP)
        if len(core) >= 2 and core.isalpha():
            cores.append(core)
    return cores


def _verdict(m: dict[str, float]) -> str:
    # The words-per-line signal only means something on a substantial document;
    # a genuinely short list would trip it spuriously.
    dense = m["n_tokens"] > 200
    if (
        m["wordlike_frac"] < 0.60
        or m["internal_caps_rate"] > 0.15
        or m["alpha_ratio"] < 0.30
        or (dense and m["words_per_line"] < 2.0)
    ):
        return "BAD"
    if (
        m["wordlike_frac"] < 0.80
        or m["internal_caps_rate"] > 0.06
        or m["long_token_frac"] > 0.02
        or m["garbage_line_frac"] > 0.50
        or (dense and m["words_per_line"] < 4.0)
    ):
        return "WARN"
    return "OK"


def score_text(text: str) -> dict[str, Any]:
    """Compute extraction-quality metrics and a verdict for one document."""
    tokens = text.split()
    n_tokens = len(tokens)
    non_space = [c for c in text if not c.isspace()]
    alpha = sum(c.isalpha() for c in non_space)
    alpha_ratio = alpha / len(non_space) if non_space else 0.0

    cores = _alpha_cores(text)
    n_cores = len(cores)
    wordlike = sum(1 for c in cores if len(c) <= 20 and _VOWEL.search(c))
    internal_caps = sum(1 for c in cores if _INTERNAL_CAPS.search(c))
    if n_cores:
        wordlike_frac = wordlike / n_cores
        internal_caps_rate = internal_caps / n_cores
    else:
        # No length>=2 alphabetic words at all. Healthy only if the document is
        # genuinely empty; if it has tokens but none form words, the text layer
        # is shredded (letter-spacing / per-glyph positioning) — flag as failure.
        wordlike_frac = 1.0 if n_tokens == 0 else 0.0
        internal_caps_rate = 0.0

    long_tokens = sum(1 for t in tokens if len(t) > 30)
    long_token_frac = long_tokens / n_tokens if n_tokens else 0.0
    mean_word_len = sum(len(t) for t in tokens) / n_tokens if n_tokens else 0.0

    lines = [ln for ln in text.splitlines() if ln.strip()]
    garbage_lines = 0
    for ln in lines:
        ns = [c for c in ln if not c.isspace()]
        if ns and sum(c.isalpha() for c in ns) / len(ns) < 0.35:
            garbage_lines += 1
    garbage_line_frac = garbage_lines / len(lines) if lines else 0.0
    words_per_line = n_tokens / len(lines) if lines else float(n_tokens)

    # Verdict is computed from raw values; rounding is display-only and applied
    # afterward so a value just under a threshold can't round past it.
    raw = {
        "n_chars": len(text),
        "n_tokens": n_tokens,
        "alpha_ratio": alpha_ratio,
        "wordlike_frac": wordlike_frac,
        "internal_caps_rate": internal_caps_rate,
        "long_token_frac": long_token_frac,
        "mean_word_len": mean_word_len,
        "garbage_line_frac": garbage_line_frac,
        "words_per_line": words_per_line,
    }
    metrics = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in raw.items()}
    metrics["verdict"] = _verdict(raw)
    return metrics


def audit_corpus(corpus_jsonl: str | Path) -> dict[str, Any]:
    """Score every document in ``corpus.jsonl``. Documents are returned
    worst-first (BAD before WARN before OK, then by interleave severity)."""
    scored: list[dict[str, Any]] = []
    with open(corpus_jsonl) as fh:
        for line in fh:
            rec = json.loads(line)
            entry = {
                "source": rec.get("source"),
                "system": rec.get("system"),
                "doc_type": rec.get("doc_type"),
                **score_text(rec["text"]),
            }
            scored.append(entry)
    scored.sort(key=lambda s: (VERDICT_ORDER[s["verdict"]], -s["internal_caps_rate"]))
    report = {
        "n_documents": len(scored),
        "verdicts": dict(Counter(s["verdict"] for s in scored)),
        "documents": scored,
    }
    logger.info("audited %d documents: %s", report["n_documents"], report["verdicts"])
    return report
