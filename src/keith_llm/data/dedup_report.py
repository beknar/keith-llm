"""Cross-source duplication report and optional automated removal.

Detects documents whose content substantially overlaps another's — including
the same adventure/book present as both a clean 5etools text render and a messy
PDF extraction, which the in-run :func:`keith_llm.data.dedup.minhash_dedup`
pass misses.

It scores pairs by the **overlap coefficient** ``|A∩B| / min(|A|, |B|)`` on
5-word shingles, not Jaccard. Containment survives large size differences: a
small stat block fully inside a big book scores ~1.0, whereas its Jaccard —
and therefore MinHash/LSH — would be small. Intersections are computed exactly
via an inverted shingle index, so no pair is missed and non-overlapping pairs
cost nothing.

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

# A shingle in more than this many documents is a common phrase / residual
# boilerplate, not a duplication signal — skipping it also caps the pairwise
# blow-up a very common shingle would cause.
_MAX_DOC_FREQUENCY = 50

_VERDICT_RANK = {"OK": 2, "WARN": 1, "BAD": 0}


def find_overlaps(
    records: list[Record], threshold: float = 0.75, k: int = 5
) -> tuple[list[tuple[int, int, float, float]], list[int], dict[tuple[int, int], int]]:
    """Return ``(pairs, sizes, shared)``.

    ``pairs`` are ``(i, j, overlap, jaccard)`` with ``i < j`` and
    ``overlap >= threshold``, highest overlap first. ``sizes[i]`` is document
    i's distinct-shingle count. ``shared`` maps every co-occurring ``(i, j)``
    to its exact shared-shingle count (kept for later lookup).
    """
    sizes: list[int] = []
    postings: dict[int, list[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        hs = shingle_hashes(rec["text"], k)
        sizes.append(len(hs))
        for h in hs:
            postings[h].append(idx)

    shared: dict[tuple[int, int], int] = defaultdict(int)
    for docs in postings.values():
        if 2 <= len(docs) <= _MAX_DOC_FREQUENCY:
            for a, b in combinations(sorted(docs), 2):
                shared[(a, b)] += 1

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


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


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


def _cluster(n: int, pairs: list[tuple[int, int, float, float]]) -> list[list[int]]:
    uf = _UnionFind(n)
    for i, j, _, _ in pairs:
        uf.union(i, j)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in {x for p in pairs for x in p[:2]}:
        groups[uf.find(i)].append(i)
    return [sorted(members) for members in groups.values()]


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
    """Group overlapping documents into clusters, choose one to keep per
    cluster, and list the rest as removal candidates. Non-destructive."""
    pairs, sizes, shared = find_overlaps(records, threshold)
    clusters_out: list[dict[str, Any]] = []
    drop_files: list[str] = []

    for members in _cluster(len(records), pairs):
        keep = max(members, key=lambda i: _keep_key(records[i]))
        drops = []
        for i in members:
            if i == keep:
                continue
            lo, hi = (i, keep) if i < keep else (keep, i)
            inter = shared.get((lo, hi), 0)
            m = min(sizes[i], sizes[keep])
            summary = _doc_summary(records[i], sizes[i])
            summary["overlap_with_keep"] = round(inter / m, 4) if m else 0.0
            drops.append(summary)
            if records[i].get("source"):
                drop_files.append(records[i]["source"])
        drops.sort(key=lambda d: d["overlap_with_keep"], reverse=True)
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
