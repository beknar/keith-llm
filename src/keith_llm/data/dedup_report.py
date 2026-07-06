"""Cross-source duplication report and optional automated removal.

Detects documents whose content substantially overlaps another's — including
the same adventure/book present as both a clean 5etools text render and a messy
PDF extraction, which the in-run :func:`keith_llm.data.dedup.minhash_dedup`
pass misses.

It scores pairs by the **overlap coefficient** ``|A∩B| / min(|A|, |B|)`` on
5-word shingles, not Jaccard. Containment survives large size differences: a
small stat block fully inside a big book scores ~1.0, whereas its Jaccard —
and therefore MinHash/LSH — would be small. Intersections are computed exactly
via an inverted shingle index, so non-overlapping pairs cost nothing. The one
exactness caveat: a shingle shared across more than ``max_doc_frequency``
documents is treated as boilerplate and skipped (and the skip is logged), which
bounds the pairwise cost but means overlaps among content duplicated across
that many documents can be undercounted.

The report is non-destructive. Removal acts on the SOURCE FILES (``corpus.jsonl``
is rebuilt from sources every ingest, so editing it is pointless) and, by
default, quarantines them so the action is reversible.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from keith_llm.data.audit import score_text
from keith_llm.data.dedup import Record, shingle_hashes

logger = logging.getLogger(__name__)

# Default cap: a shingle in more than this many documents is a common phrase /
# residual boilerplate, not a specific-duplicate signal — skipping it also caps
# the pairwise blow-up a very common shingle would cause. A given adventure/book
# realistically appears in only a handful of sources, well under this.
_MAX_DOC_FREQUENCY = 50

_VERDICT_RANK = {"OK": 2, "WARN": 1, "BAD": 0}


def find_overlaps(
    records: list[Record],
    threshold: float = 0.75,
    k: int = 5,
    max_doc_frequency: int = _MAX_DOC_FREQUENCY,
) -> tuple[list[tuple[int, int, float, float]], list[int], dict[tuple[int, int], int]]:
    """Return ``(pairs, sizes, shared)``.

    ``pairs`` are ``(i, j, overlap, jaccard)`` with ``i < j`` and
    ``overlap >= threshold``, highest overlap first. ``sizes[i]`` is document
    i's distinct-shingle count. ``shared`` maps every co-occurring ``(i, j)``
    to its shared-shingle count.

    Shingles shared across more than ``max_doc_frequency`` documents are treated
    as boilerplate and skipped (logged) to bound cost; raise it if you expect
    genuine content duplicated across that many sources.
    """
    sizes: list[int] = []
    postings: dict[int, list[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        hs = shingle_hashes(rec["text"], k)
        sizes.append(len(hs))
        for h in hs:
            postings[h].append(idx)

    shared: dict[tuple[int, int], int] = defaultdict(int)
    skipped = 0
    for docs in postings.values():
        if len(docs) < 2:
            continue
        if len(docs) > max_doc_frequency:
            skipped += 1
            continue
        for a, b in combinations(sorted(docs), 2):
            shared[(a, b)] += 1
    if skipped:
        logger.warning(
            "%d shingles skipped as boilerplate (shared across > %d documents); "
            "overlaps among that-heavily-duplicated content may be undercounted — "
            "raise max_doc_frequency to include them",
            skipped,
            max_doc_frequency,
        )

    pairs = []
    for (i, j), inter in shared.items():
        m = min(sizes[i], sizes[j])
        if not m:
            continue
        overlap = inter / m
        if overlap >= threshold:
            union = sizes[i] + sizes[j] - inter
            pairs.append((i, j, overlap, inter / union if union else 0.0))
    pairs.sort(key=lambda p: p[2], reverse=True)
    return pairs, sizes, dict(shared)


def _keep_key(rec: Record) -> tuple:
    """Higher sorts as the one to KEEP: better extraction, non-PDF, longer,
    then a deterministic path tiebreak."""
    s = score_text(rec["text"])
    is_pdf = Path(rec.get("source") or "").suffix.lower() == ".pdf"
    return (
        _VERDICT_RANK.get(s["verdict"], 0),
        s["wordlike_frac"],
        not is_pdf,
        len(rec["text"]),
        rec.get("source") or "",
    )


def _doc_summary(rec: Record, size: int) -> dict[str, Any]:
    return {
        "source": rec.get("source"),
        "system": rec.get("system"),
        "doc_type": rec.get("doc_type"),
        "n_chars": len(rec["text"]),
        "n_shingles": size,
        "verdict": score_text(rec["text"])["verdict"],
    }


def build_report(records: list[Record], threshold: float = 0.75) -> dict[str, Any]:
    """Greedily keep the best copy of each duplicate group and list the rest as
    removal candidates. Best-first: a document is only dropped when it overlaps
    an already-KEPT document by >= threshold, so nothing is deleted merely for
    resembling another doc that was itself dropped. Non-destructive."""
    pairs, sizes, _ = find_overlaps(records, threshold)
    neighbors: dict[int, dict[int, float]] = defaultdict(dict)
    for i, j, overlap, _ in pairs:
        neighbors[i][j] = overlap
        neighbors[j][i] = overlap

    order = sorted(range(len(records)), key=lambda i: _keep_key(records[i]), reverse=True)
    removed: set[int] = set()
    clusters_out: list[dict[str, Any]] = []
    drop_files: list[str] = []

    for keep in order:
        if keep in removed:
            continue
        drops = []
        for other, overlap in sorted(neighbors[keep].items(), key=lambda kv: kv[1], reverse=True):
            if other in removed:
                continue
            removed.add(other)
            summary = _doc_summary(records[other], sizes[other])
            summary["overlap_with_keep"] = round(overlap, 4)
            drops.append(summary)
            if records[other].get("source"):
                drop_files.append(records[other]["source"])
        if drops:
            clusters_out.append({"keep": _doc_summary(records[keep], sizes[keep]), "drop": drops})

    clusters_out.sort(
        key=lambda c: max((d["overlap_with_keep"] for d in c["drop"]), default=0.0),
        reverse=True,
    )
    return {
        "threshold": threshold,
        "n_documents": len(records),
        "n_clusters": len(clusters_out),
        "n_drop_files": len(drop_files),
        "clusters": clusters_out,
        "drop_files": sorted(set(drop_files)),
    }


def report_corpus(corpus_jsonl: str | Path, threshold: float = 0.75) -> dict[str, Any]:
    with open(corpus_jsonl) as fh:
        records = [json.loads(line) for line in fh]
    report = build_report(records, threshold)
    logger.info(
        "duplication report: %d clusters, %d files flagged (threshold %.2f)",
        report["n_clusters"],
        report["n_drop_files"],
        threshold,
    )
    return report


def apply_removals(
    drop_files: list[str],
    root: str | Path = ".",
    hard: bool = False,
    quarantine_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Remove the flagged source files. By default they are moved (reversible)
    into ``data/quarantine/`` preserving their relative path; ``hard=True``
    deletes them permanently. Missing files are reported, not fatal."""
    root = Path(root)
    quarantine = Path(quarantine_dir) if quarantine_dir else root / "data" / "quarantine"
    removed: list[str] = []
    missing: list[str] = []
    for rel in drop_files:
        src = root / rel
        if not src.exists():
            missing.append(rel)
            continue
        if hard:
            src.unlink()
        else:
            dest = quarantine / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
        removed.append(rel)
    logger.info("removed %d files (hard=%s), %d missing", len(removed), hard, len(missing))
    return {
        "removed": removed,
        "missing": missing,
        "hard": hard,
        "quarantine": None if hard else str(quarantine),
    }
