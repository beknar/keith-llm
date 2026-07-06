"""Content-addressed cache of extracted+cleaned document text.

``build_corpus`` rebuilds the whole corpus every run, re-extracting (and
re-OCR-ing) every source file — expensive when the corpus has scanned PDFs.
This cache keys the cleaned text by ``sha256(file bytes)`` plus an extraction
version, so on re-ingest an unchanged file is served from the cache instead of
being extracted again. Content addressing means the cache also survives file
renames/moves and shares one entry across byte-identical copies.

The version tag captures everything that changes a file's extracted text
besides its bytes: this repo's extraction/cleaning logic (``EXTRACT_VERSION``,
bump it when that changes), whether OCR was applied, and the installed versions
of the extraction/OCR tools (pypdf, pdfplumber, pytesseract, and the Tesseract
engine). So upgrading — not just installing — any of those automatically
invalidates stale entries; no manual discipline beyond bumping
``EXTRACT_VERSION`` for in-repo changes. ``(hash, version)`` is the primary key,
so with-OCR and without-OCR (`--no-ocr`) extractions of the same file coexist
rather than clobbering each other.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

logger = logging.getLogger(__name__)

# Bump when extraction or cleaning logic changes in a way that alters output.
EXTRACT_VERSION = "1"


def _dep_version(name: str) -> str:
    try:
        return _pkg_version(name)
    except PackageNotFoundError:
        return "none"


def current_version(applied_ocr: bool) -> str:
    """A cache key component that changes whenever extraction output could —
    repo logic, OCR-applied flag, and the versions of the extraction/OCR tools."""
    parts = [
        EXTRACT_VERSION,
        f"pypdf={_dep_version('pypdf')}",
        f"pdfplumber={_dep_version('pdfplumber')}",
        f"ocr={int(applied_ocr)}",
    ]
    if applied_ocr:
        parts.append(f"pypdfium2={_dep_version('pypdfium2')}")
        parts.append(f"pytesseract={_dep_version('pytesseract')}")
        try:
            import pytesseract

            parts.append(f"tesseract={pytesseract.get_tesseract_version()}")
        except Exception:  # noqa: BLE001 - engine version is best-effort
            parts.append("tesseract=?")
    return "|".join(parts)


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
