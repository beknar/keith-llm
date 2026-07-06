"""Deduplication passes: exact, cross-document boilerplate, MinHash near-dup.

All hashing is seeded/deterministic so a given corpus always dedups the same
way regardless of PYTHONHASHSEED.
"""

from __future__ import annotations

import hashlib
import itertools
import random
import re
from collections import Counter, defaultdict
from typing import Any

Record = dict[str, Any]

_WS = re.compile(r"\s+")
_MERSENNE = (1 << 61) - 1


def _norm(text: str) -> str:
    return _WS.sub(" ", text.lower()).strip()


def exact_dedup(records: list[Record]) -> list[Record]:
    seen: set[str] = set()
    out = []
    for rec in records:
        digest = hashlib.sha1(_norm(rec["text"]).encode()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            out.append(rec)
    return out


def paragraph_dedup(
    records: list[Record], max_docs: int = 3, min_para_chars: int = 80
) -> list[Record]:
    """Remove paragraphs that appear in more than ``max_docs`` documents —
    e.g. the OGL legal text repeated in every OGL book. Short paragraphs
    (headings, dice lines) are exempt; documents left empty are dropped.
    """
    para_docs: Counter[str] = Counter()
    for rec in records:
        paras = {_norm(p) for p in rec["text"].split("\n\n") if len(p.strip()) >= min_para_chars}
        para_docs.update(paras)
    boiler = {p for p, n in para_docs.items() if n > max_docs}
    out = []
    for rec in records:
        kept = [
            p
            for p in rec["text"].split("\n\n")
            if len(p.strip()) < min_para_chars or _norm(p) not in boiler
        ]
        text = "\n\n".join(kept).strip()
        if text:
            out.append({**rec, "text": text})
    return out


def shingle_hashes(text: str, k: int = 5) -> set[int]:
    """Deterministic 64-bit hashes of the text's distinct k-word shingles.

    Shared by MinHash signing here and by the overlap report, so both use one
    shingling definition.
    """
    words = _norm(text).split()
    if not words:
        return set()
    if len(words) < k:
        grams = {" ".join(words)}
    else:
        grams = {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}
    return {
        int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big") for g in grams
    }


def _signature(hashes: set[int], perms: list[tuple[int, int]]) -> list[int]:
    return [min((a * h + b) % _MERSENNE for h in hashes) for a, b in perms]


def minhash_dedup(
    records: list[Record],
    threshold: float = 0.85,
    num_hashes: int = 64,
    bands: int = 16,
) -> list[Record]:
    """Drop near-duplicate documents (estimated Jaccard >= threshold on 5-word
    shingles), keeping the longer of each pair. LSH banding keeps comparisons
    to candidate pairs only."""
    rng = random.Random(42)
    perms = [(rng.randrange(1, _MERSENNE), rng.randrange(_MERSENNE)) for _ in range(num_hashes)]
    sigs: list[list[int] | None] = []
    for rec in records:
        hs = shingle_hashes(rec["text"])
        sigs.append(_signature(hs, perms) if hs else None)

    rows = num_hashes // bands
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    for i, sig in enumerate(sigs):
        if sig is None:
            continue
        for band in range(bands):
            buckets[(band, tuple(sig[band * rows : (band + 1) * rows]))].append(i)

    drop: set[int] = set()
    for idxs in buckets.values():
        for i, j in itertools.combinations(idxs, 2):
            if i in drop or j in drop:
                continue
            sim = sum(x == y for x, y in zip(sigs[i], sigs[j], strict=True)) / num_hashes
            if sim >= threshold:
                loser = i if len(records[i]["text"]) < len(records[j]["text"]) else j
                drop.add(loser)
    return [rec for k, rec in enumerate(records) if k not in drop]
