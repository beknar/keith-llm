"""Manifest loading and corpus orchestration: files -> corpus.jsonl."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from keith_llm.constants import DOC_TYPES, SYSTEMS
from keith_llm.data.archive import extracted_documents, is_archive
from keith_llm.data.clean import clean_pages, clean_text, is_quality
from keith_llm.data.dedup import exact_dedup, minhash_dedup, paragraph_dedup
from keith_llm.data.ingest import SUPPORTED_EXTS, extract_pages

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = {"glob", "system", "doc_type", "license", "publishable"}


@dataclass
class SourceSpec:
    glob: str
    system: str
    doc_type: str
    license: str
    publishable: bool


def load_manifest(path: str | Path) -> list[SourceSpec]:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    specs = []
    for i, entry in enumerate(raw.get("sources", [])):
        missing = _REQUIRED_KEYS - entry.keys()
        if missing:
            raise ValueError(f"sources[{i}] missing keys: {sorted(missing)}")
        unknown = entry.keys() - _REQUIRED_KEYS
        if unknown:
            raise ValueError(f"sources[{i}] unknown keys: {sorted(unknown)}")
        if entry["system"] not in SYSTEMS:
            raise ValueError(f"sources[{i}]: unknown system {entry['system']!r}")
        if entry["doc_type"] not in DOC_TYPES:
            raise ValueError(f"sources[{i}]: unknown doc_type {entry['doc_type']!r}")
        specs.append(SourceSpec(**entry))
    return specs


def filter_publishable(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if r["publishable"]]


def build_corpus(
    manifest_path: str | Path,
    out_path: str | Path,
    root: str | Path = ".",
    enable_ocr: bool = True,
    use_cache: bool = True,
    cache_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the full pipeline (extract -> clean -> quality filter -> dedup) and
    write ``out_path`` as JSONL. Returns summary stats. ``enable_ocr`` OCRs
    image-only PDF pages when the ``ocr`` extra is installed. ``use_cache``
    serves unchanged files' cleaned text from a content-addressed cache so
    re-ingest skips re-extraction/OCR."""
    from keith_llm.data.extract_cache import ExtractionCache, current_version, hash_file
    from keith_llm.data.ocr import ocr_available

    root = Path(root)
    specs = load_manifest(manifest_path)
    records: list[dict[str, Any]] = []
    counters = {"files_scanned": 0, "dropped_low_quality": 0, "cache_hits": 0}

    cache: ExtractionCache | None = None
    if use_cache:
        if cache_path is None:
            cache_path = root / "data" / "cache" / "extraction.sqlite"
        try:
            applied_ocr = enable_ocr and ocr_available()
            cache = ExtractionCache(cache_path, current_version(applied_ocr))
        except Exception as exc:  # noqa: BLE001 - a bad cache must not block ingestion
            logger.warning("extraction cache unavailable (%s); proceeding without it", exc)
            cache = None

    def ingest_unit(source: str, filepath: Path, spec: SourceSpec) -> None:
        counters["files_scanned"] += 1
        content_hash = None
        text = None
        if cache is not None:
            try:
                content_hash = hash_file(filepath)
            except Exception as exc:  # noqa: BLE001 - unhashable file -> extract, don't cache
                logger.warning("cache hash failed for %s: %s", source, exc)
            if content_hash is not None:
                try:
                    text = cache.get(content_hash)
                except Exception as exc:  # noqa: BLE001 - read failure -> extract; put still works
                    logger.warning("cache read failed for %s: %s", source, exc)
        if text is not None:
            counters["cache_hits"] += 1
        else:
            try:
                pages = extract_pages(filepath, enable_ocr=enable_ocr)
            except Exception as exc:  # noqa: BLE001 - one bad file must not kill the run
                logger.warning("skipping %s: %s", source, exc)
                return
            text = clean_text(clean_pages(pages))
            if cache is not None and content_hash is not None:
                try:
                    cache.put(content_hash, text)
                except Exception as exc:  # noqa: BLE001 - failing to cache is not fatal
                    logger.warning("cache write failed for %s: %s", source, exc)
        if not is_quality(text):
            counters["dropped_low_quality"] += 1
            logger.info("low quality, skipping: %s", source)
            return
        records.append(
            {
                "source": source,
                "system": spec.system,
                "doc_type": spec.doc_type,
                "license": spec.license,
                "publishable": spec.publishable,
                "text": text,
            }
        )

    archives_expanded = 0
    skipped_unsupported = 0
    try:
        for spec in specs:
            for path in sorted(root.glob(spec.glob)):
                if not path.is_file():
                    continue
                rel = str(path.relative_to(root))
                if path.suffix.lower() in SUPPORTED_EXTS:
                    ingest_unit(rel, path, spec)
                elif is_archive(path):
                    archives_expanded += 1
                    try:
                        with extracted_documents(path, SUPPORTED_EXTS) as members:
                            if not members:
                                logger.info("archive contained no ingestible files: %s", rel)
                            for name, mpath in members:
                                ingest_unit(f"{rel}!{name}", mpath, spec)
                    except Exception as exc:  # noqa: BLE001 - a bad archive must not kill the run
                        logger.warning("skipping unreadable archive %s: %s", rel, exc)
                else:
                    skipped_unsupported += 1
                    logger.warning("skipping unsupported file type: %s", rel)
    finally:
        if cache is not None:
            cache.close()

    n_extracted = len(records)
    records = exact_dedup(records)
    n_after_exact = len(records)
    records = paragraph_dedup(records)
    records = minhash_dedup(records)

    for rec in records:
        rec["id"] = hashlib.sha1(rec["text"].encode()).hexdigest()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats = {
        "files_scanned": counters["files_scanned"],
        "cache_hits": counters["cache_hits"],
        "archives_expanded": archives_expanded,
        "skipped_unsupported": skipped_unsupported,
        "dropped_low_quality": counters["dropped_low_quality"],
        "dropped_exact_dup": n_extracted - n_after_exact,
        "dropped_near_dup": n_after_exact - len(records),
        "documents": len(records),
        "characters": sum(len(r["text"]) for r in records),
        "by_system": dict(Counter(r["system"] for r in records)),
    }
    logger.info("corpus written to %s: %s", out_path, stats)
    return stats
