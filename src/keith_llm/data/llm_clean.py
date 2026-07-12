"""Local-LLM clean pass for extraction-damaged corpus documents.

Some PDFs extract badly — words run together, columns interleave, OCR leaves
garbage. :mod:`keith_llm.data.audit` flags these (verdict ``WARN``/``BAD``);
this module hands the flagged text to a local model and asks it to *repair*
the extraction into clean single-column prose.

The rewrite is trusted only when it passes two gates, so the model can never
quietly inject hallucinated facts into the training corpus:

1. **Improvement** — the cleaned text must score better on the same audit
   metrics (better verdict, or a measurable gain in wordlike/interleave rates).
2. **Faithfulness** — the cleaned text's letters must overlap the original's
   almost entirely (character n-gram containment). Fixing spacing/columns keeps
   the letters; inventing sentences introduces new ones, which fails the gate.

A document that fails either gate keeps its original text (or is dropped, if it
was ``BAD`` and ``drop_failed`` is set). Only audit-flagged documents are ever
sent to the model — clean documents pass through untouched and unread.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from keith_llm.data.audit import VERDICT_ORDER, score_text
from keith_llm.llm import DEFAULT_BASE_URL, OllamaClient

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You repair text that was extracted from a PDF. You fix formatting only; "
    "you never change, summarize, or add to the content. Output only the "
    "repaired text — no preamble, no commentary, no markdown fences."
)

_CLEAN_TMPL = (
    "The following text was extracted from a PDF and is damaged: words are run "
    "together, columns are interleaved, spacing and line breaks are wrong, and "
    "there may be OCR garbage.\n\n"
    "Rewrite it as clean, readable single-column prose. Rules:\n"
    "- Fix spacing, join split words, separate run-together words, and reflow "
    "lines into normal paragraphs.\n"
    "- Preserve the original wording, facts, numbers, names, and stat blocks "
    "EXACTLY. Do NOT summarize, paraphrase, translate, add, or invent anything.\n"
    "- Drop only obvious junk: page numbers, repeated running headers, and "
    "garbled non-words.\n"
    "- Output ONLY the repaired text.\n\n"
    "TEXT:\n{text}"
)


def _chunks(text: str, max_chars: int) -> list[str]:
    """Split ``text`` into <= ``max_chars`` pieces on paragraph boundaries, hard-
    splitting any single paragraph that is itself over budget."""
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    cur: list[str] = []
    size = 0
    for para in text.split("\n\n"):
        piece = para + "\n\n"
        while len(piece) > max_chars:
            # a single oversize paragraph: emit what we have, then hard-split it
            if cur:
                chunks.append("".join(cur))
                cur, size = [], 0
            chunks.append(piece[:max_chars])
            piece = piece[max_chars:]
        if size + len(piece) > max_chars and cur:
            chunks.append("".join(cur))
            cur, size = [], 0
        cur.append(piece)
        size += len(piece)
    if cur:
        chunks.append("".join(cur))
    return chunks


def _strip_fences(text: str) -> str:
    """Drop a wrapping ```...``` code fence if the model added one anyway."""
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
    return "\n".join(lines).strip()


def clean_text(client: OllamaClient, text: str, max_chars: int = 6000) -> str:
    """Repair one document's text with the model, chunk by chunk. Raises if the
    model call fails (the caller decides whether to keep or drop the original)."""
    out = []
    for chunk in _chunks(text, max_chars):
        resp = client.chat(_CLEAN_TMPL.format(text=chunk), system=_SYSTEM)
        out.append(_strip_fences(resp))
    return "\n\n".join(out).strip()


def _letter_ngrams(text: str, n: int = 4) -> set[str]:
    letters = "".join(c.lower() for c in text if c.isalpha())
    return {letters[i : i + n] for i in range(len(letters) - n + 1)}


def faithfulness(original: str, cleaned: str) -> float:
    """Fraction of the cleaned text's letter 4-grams that also occur in the
    original. ~1.0 when only spacing/columns changed; drops as the model
    introduces letters that weren't there (i.e. invents content). Space- and
    split-tolerant because spaces and punctuation are ignored."""
    cleaned_grams = _letter_ngrams(cleaned)
    if not cleaned_grams:
        return 0.0
    original_grams = _letter_ngrams(original)
    return len(cleaned_grams & original_grams) / len(cleaned_grams)


def _improved(old: dict[str, Any], new: dict[str, Any]) -> bool:
    """True if ``new`` is a genuinely cleaner extraction than ``old``."""
    ov, nv = VERDICT_ORDER[old["verdict"]], VERDICT_ORDER[new["verdict"]]
    if nv != ov:
        return nv > ov  # higher verdict rank (OK=2) is better
    # same verdict: demand a measurable drop in the interleave/gibberish signals
    return (
        new["wordlike_frac"] >= old["wordlike_frac"] + 0.02
        and new["internal_caps_rate"] <= old["internal_caps_rate"]
    )


def clean_corpus(
    corpus_jsonl: str | Path,
    out_path: str | Path,
    *,
    model: str = "gpt-oss",
    ollama_url: str = DEFAULT_BASE_URL,
    target_verdicts: tuple[str, ...] = ("BAD", "WARN"),
    min_overlap: float = 0.80,
    drop_failed: bool = False,
    max_chars: int = 6000,
    max_docs: int | None = None,
    client: OllamaClient | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Repair audit-flagged documents in ``corpus_jsonl`` with a local LLM and
    write the result to ``out_path``.

    Only documents whose audit verdict is in ``target_verdicts`` are sent to the
    model. A rewrite replaces the original only if it both improves the audit
    score and stays faithful (letter overlap >= ``min_overlap``); otherwise the
    original is kept, or dropped when ``drop_failed`` and it was ``BAD``.
    ``max_docs`` caps how many flagged documents are actually processed (for a
    cheap trial run); the rest pass through untouched.
    """
    records = []
    with open(corpus_jsonl) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if client is None:
        client = OllamaClient(model=model, base_url=ollama_url)
        if not client.available():
            raise SystemExit(
                f"ollama not reachable at {ollama_url}; start it or point --ollama-url at the host"
            )

    counts = {"targeted": 0, "replaced": 0, "kept": 0, "dropped": 0, "unfaithful": 0}
    report: list[dict[str, Any]] = []
    out_records: list[dict[str, Any]] = []
    processed = 0

    for rec in records:
        old = score_text(rec["text"])
        if old["verdict"] not in target_verdicts:
            out_records.append(rec)
            continue
        if max_docs is not None and processed >= max_docs:
            out_records.append(rec)  # over the trial cap: leave untouched
            continue

        processed += 1
        counts["targeted"] += 1
        source = rec.get("source")

        try:
            cleaned = clean_text(client, rec["text"], max_chars)
        except Exception as exc:  # noqa: BLE001 - a failed rewrite must not kill the run
            logger.warning("clean failed for %s: %s", source, exc)
            out_records.append(rec)
            counts["kept"] += 1
            report.append(
                {
                    "source": source,
                    "old": old["verdict"],
                    "action": "keep",
                    "reason": "clean_failed",
                }
            )
            continue

        new = score_text(cleaned)
        overlap = faithfulness(rec["text"], cleaned)
        faithful = overlap >= min_overlap and bool(cleaned)
        improved = _improved(old, new)

        if faithful and improved:
            newrec = {**rec, "text": cleaned, "cleaned": True}
            newrec["id"] = hashlib.sha1(cleaned.encode()).hexdigest()
            out_records.append(newrec)
            counts["replaced"] += 1
            action, reason = "replace", "improved+faithful"
        else:
            reason = "not_improved" if faithful else "unfaithful"
            if not faithful:
                counts["unfaithful"] += 1
            if drop_failed and old["verdict"] == "BAD":
                counts["dropped"] += 1
                action = "drop"
            else:
                out_records.append(rec)
                counts["kept"] += 1
                action = "keep"

        report.append(
            {
                "source": source,
                "old": old["verdict"],
                "new": new["verdict"],
                "overlap": round(overlap, 3),
                "action": action,
                "reason": reason,
            }
        )

    if not dry_run:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as fh:
            for rec in out_records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats = {
        "n_documents": len(records),
        "documents_out": len(out_records),
        **counts,
        "dry_run": dry_run,
        "report": report,
    }
    logger.info("llm clean pass: %s", {k: v for k, v in stats.items() if k != "report"})
    return stats
