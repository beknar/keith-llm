"""Content-addressed cache of extracted+cleaned document text.

``build_corpus`` rebuilds the whole corpus every run, re-extracting (and
re-OCR-ing) every source file — expensive when the corpus has scanned PDFs.
This cache keys the cleaned text by ``sha256(file bytes)`` plus an extraction
version, so on re-ingest an unchanged file is served from the cache instead of
being extracted again. Content addressing means the cache also survives file
renames/moves and shares one entry across byte-identical copies.

The version tag encodes both the extraction/cleaning logic (``EXTRACT_VERSION``,
bump it when that changes) and whether OCR was applied, so installing Tesseract
— or changing the pipeline — correctly invalidates stale entries. ``(hash,
version)`` is the primary key, so with-OCR and without-OCR (`--no-ocr`)
extractions of the same file coexist rather than clobbering each other.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

# Bump when extraction or cleaning logic changes in a way that alters output.
EXTRACT_VERSION = "1"


def current_version(applied_ocr: bool) -> str:
    return f"{EXTRACT_VERSION}:ocr={int(applied_ocr)}"


def hash_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


class ExtractionCache:
    """SQLite-backed store of ``(content_hash, version) -> cleaned_text``."""

    def __init__(self, path: str | Path, version: str):
        self.version = version
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS entries "
            "(hash TEXT NOT NULL, version TEXT NOT NULL, text TEXT NOT NULL, "
            "PRIMARY KEY (hash, version))"
        )
        self.conn.commit()

    def get(self, content_hash: str) -> str | None:
        row = self.conn.execute(
            "SELECT text FROM entries WHERE hash = ? AND version = ?",
            (content_hash, self.version),
        ).fetchone()
        return row[0] if row else None

    def put(self, content_hash: str, text: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO entries (hash, version, text) VALUES (?, ?, ?)",
            (content_hash, self.version, text),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
