"""Convert a tree of messy source files into clean, readable training text.

Walks a directory (recursing into subdirectories and uncompressing archives)
and turns each PDF / HTML / image / text file into readable plain text, written
mirror-structured under an output directory. Unreadable results — gibberish
OCR of a decorative image, a shredded scan, a file that never had text — are
discarded by a readability gate rather than written.

It reuses the ingest building blocks (column-aware PDF extraction with OCR
fallback, ftfy/NFKC/​de-hyphenation cleaning, the audit gibberish scorer, safe
archive extraction) and adds: HTML→text, standalone image→OCR, a full-document
OCR fallback for garbled PDF text layers, line reflow (joins wrapped lines into
paragraphs), and clumped-word repair (splits run-together tokens).

Optional capabilities degrade gracefully if their deps are missing:
``pip install -e ".[convert]"`` (+ the system ``tesseract`` binary for OCR).
"""

from __future__ import annotations

import logging
import os
import re
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from keith_llm.data.archive import _SINGLE_OPENERS, is_archive
from keith_llm.data.audit import score_text
from keith_llm.data.clean import clean_pages, clean_text, is_quality
from keith_llm.data.docs import DOC_EXTS, extract_document
from keith_llm.data.ocr import ocr_available
from keith_llm.data.pdf_layout import extract_pdf_pages

logger = logging.getLogger(__name__)

PDF_EXTS = {".pdf"}
HTML_EXTS = {".html", ".htm", ".xhtml"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
TEXT_EXTS = {".txt", ".md", ".markdown"}
# DOC_EXTS (.docx/.doc/.odt/.rtf/.epub/.mobi) are already-digital documents —
# clean text, no OCR — handled by keith_llm.data.docs.
CONVERTIBLE_EXTS = PDF_EXTS | HTML_EXTS | IMAGE_EXTS | TEXT_EXTS | DOC_EXTS

_LONG_TOKEN = re.compile(r"[A-Za-z]{19,}")  # candidate clumped-word runs
_BLANKLINE = re.compile(r"\n\s*\n")

_MAX_DEPTH = 40  # dir/archive nesting cap — stops symlink loops and archive quines
_MAX_ARCHIVE_BYTES = 4 * 1024**3  # 4 GB uncompressed per archive (bomb guard)
_MAX_ARCHIVE_MEMBERS = 20_000


def _is_junk(name: str) -> bool:
    """OS metadata cruft that isn't real content — e.g. macOS AppleDouble
    ``._foo`` resource forks and the ``__MACOSX`` dir that fill Mac-made zips,
    plus ``.DS_Store`` / ``Thumbs.db``. These carry real extensions but no data."""
    return name == "__MACOSX" or name.startswith("._") or name in {".DS_Store", "Thumbs.db"}


# --- HTML ---


def html_to_text(html: str) -> str:
    """Strip HTML to text. Uses BeautifulSoup if available (robust on messy
    markup), else a stdlib parser fallback."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "head", "nav", "footer"]):
            tag.decompose()
        return soup.get_text("\n")
    except ImportError:
        from html.parser import HTMLParser

        class _Strip(HTMLParser):
            def __init__(self):
                super().__init__()
                self.parts: list[str] = []
                self._skip = 0

            def handle_starttag(self, tag, attrs):
                if tag in ("script", "style", "head"):
                    self._skip += 1

            def handle_endtag(self, tag):
                if tag in ("script", "style", "head") and self._skip:
                    self._skip -= 1

            def handle_data(self, data):
                if not self._skip and data.strip():
                    self.parts.append(data)

        p = _Strip()
        p.feed(html)
        return "\n".join(p.parts)


# --- readability post-processing ---


def reflow(text: str) -> str:
    """Join wrapped lines within each blank-line-delimited paragraph, fixing the
    extra line breaks PDF extraction leaves mid-sentence."""
    blocks = _BLANKLINE.split(text)
    out = []
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if lines:
            out.append(" ".join(lines))
    return "\n\n".join(out)


def fix_spacing(text: str) -> str:
    """Split run-together tokens ("thegoblinattacks") back into words using a
    word-frequency model, but only on long pure-alpha runs, and only when the
    split cleanly reconstructs the original — so normal prose is untouched."""
    try:
        import wordninja
    except ImportError:
        return text

    def repl(match: re.Match[str]) -> str:
        tok = match.group(0)
        parts = wordninja.split(tok)
        if len(parts) > 1 and all(len(p) >= 2 for p in parts) and "".join(parts) == tok.lower():
            return " ".join(parts)
        return tok

    return _LONG_TOKEN.sub(repl, text)


def is_readable(text: str, min_chars: int = 200) -> bool:
    """True if the text is worth keeping: long enough and not gibberish. Uses
    the audit scorer (wordlike fraction / internal-caps / words-per-line), so
    OCR noise, garbled extractions, and no-text images are rejected."""
    if not is_quality(text, min_chars=min_chars):
        return False
    return score_text(text)["verdict"] != "BAD"


# --- per-file conversion ---


def _pdf_to_text(path: Path, enable_ocr: bool, min_chars: int) -> str:
    pages = extract_pdf_pages(str(path), enable_ocr=enable_ocr)
    text = clean_text(clean_pages(pages))
    # Garbled text-layer fallback: re-OCR the whole PDF if the extraction is
    # gibberish (a broken CID font extracts junk but renders fine).
    if enable_ocr and ocr_available() and score_text(text)["verdict"] == "BAD":
        from keith_llm.data.ocr import ocr_pdf

        logger.info("PDF text layer looks garbled; re-OCRing %s", path.name)
        ocr_text = clean_text(ocr_pdf(str(path)))
        if is_readable(ocr_text, min_chars):  # only swap in OCR if it's actually readable
            text = ocr_text
    return text


def convert_file(
    path: Path,
    enable_ocr: bool = True,
    do_reflow: bool = True,
    do_fix_spacing: bool = True,
    min_chars: int = 200,
) -> str | None:
    """Convert one file to clean text, or None if it can't be read or the result
    is gibberish (discarded)."""
    suffix = path.suffix.lower()
    if suffix in PDF_EXTS:
        raw = _pdf_to_text(path, enable_ocr, min_chars)
    elif suffix in HTML_EXTS:
        raw = clean_text(html_to_text(path.read_text(encoding="utf-8", errors="replace")))
    elif suffix in IMAGE_EXTS:
        if not (enable_ocr and ocr_available()):
            return None
        from keith_llm.data.ocr import ocr_image

        raw = clean_text(ocr_image(str(path)))
    elif suffix in TEXT_EXTS:
        raw = clean_text(path.read_text(encoding="utf-8", errors="replace"))
    elif suffix in DOC_EXTS:
        extracted = extract_document(path)
        if extracted is None:  # unsupported lib/tool missing, or extraction failed
            return None
        raw = clean_text(extracted)
    else:
        return None

    text = raw
    if do_reflow:
        text = reflow(text)
    if do_fix_spacing:
        text = fix_spacing(text)
    return text if is_readable(text, min_chars) else None


# --- archives ---


def _check_budget(total_bytes: int, count: int, name: str) -> None:
    """Reject a decompression bomb before extracting."""
    if total_bytes > _MAX_ARCHIVE_BYTES:
        raise ValueError(f"{name} inflates to {total_bytes} bytes (> {_MAX_ARCHIVE_BYTES} cap)")
    if count > _MAX_ARCHIVE_MEMBERS:
        raise ValueError(f"{name} has {count} members (> {_MAX_ARCHIVE_MEMBERS} cap)")


def _copy_capped(src, dst, cap: int) -> None:
    written = 0
    while chunk := src.read(1 << 20):
        written += len(chunk)
        if written > cap:
            raise ValueError(f"compressed stream exceeds {cap}-byte cap")
        dst.write(chunk)


def _extract_archive(path: Path, dest: Path) -> None:
    """Safely extract an archive into ``dest``: traversal-guarded (modern
    zipfile/tarfile sanitize member paths) and bomb-guarded (size/member caps)."""
    suffix = path.suffix.lower()
    suffixes = [s.lower() for s in path.suffixes]
    is_tar = suffix in {".tar", ".tgz", ".tbz2", ".tbz", ".txz"} or (
        len(suffixes) >= 2 and suffixes[-2] == ".tar"
    )
    if suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
            _check_budget(sum(i.file_size for i in infos), len(infos), path.name)
            zf.extractall(dest)  # zipfile sanitizes member paths
    elif is_tar:
        with tarfile.open(path) as tf:
            members = tf.getmembers()
            _check_budget(sum(m.size for m in members), len(members), path.name)
            tf.extractall(dest, filter="data")  # rejects traversal/special files
    elif suffix in _SINGLE_OPENERS:
        inner = dest / path.name[: -len(path.suffix)]
        with _SINGLE_OPENERS[suffix](path, "rb") as src, inner.open("wb") as out:
            _copy_capped(src, out, _MAX_ARCHIVE_BYTES)
        # Stamp the extracted member with the archive's mtime (zip/tar already
        # preserve member mtimes) so a re-run's _is_fresh check can cache it
        # instead of re-extracting and re-OCRing every time.
        mtime = path.stat().st_mtime
        os.utime(inner, (mtime, mtime))


# --- tree walk ---


def _is_fresh(dest: Path, src: Path) -> bool:
    """True if a non-empty conversion of ``src`` already exists at ``dest`` and is
    at least as new as ``src``. Lets a re-run skip the expensive re-extract/OCR
    for files finished by an earlier (e.g. killed) run, while still re-converting
    any source edited since. Archive members work too: they carry the archive's
    stored/stamped mtime, older than the first run's output."""
    # `>=` (not `>`) so a source and its output landing in the same coarse mtime
    # tick still count as fresh; the cost is that a source edited *during* its own
    # conversion (same tick as the dest write) could be skipped next run — rare,
    # and --force clears it. Outputs are written atomically, so a non-empty .txt
    # is always a complete conversion, never a truncated partial.
    try:
        d = dest.stat()
        return d.st_size > 0 and d.st_mtime >= src.stat().st_mtime
    except OSError:
        return False


def convert_tree(
    src: str | Path,
    out: str | Path,
    enable_ocr: bool = True,
    do_reflow: bool = True,
    do_fix_spacing: bool = True,
    min_chars: int = 200,
    force: bool = False,
) -> dict[str, Any]:
    """Convert every convertible file under ``src`` into readable ``.txt`` files
    mirrored under ``out``. Recurses subdirectories and archives; discards
    unreadable results. Returns summary stats.

    Resumable: a file whose ``.txt`` output already exists and is newer than the
    source is skipped (counted as ``cached``), so a killed run can be restarted
    cheaply — including files inside archives. Pass ``force`` to reconvert
    everything."""
    src, out = Path(src).resolve(), Path(out)
    if not src.is_dir():
        raise NotADirectoryError(f"--src is not a directory: {src}")
    stats = {"converted": 0, "cached": 0, "rejected": 0, "skipped": 0, "archives": 0, "failed": 0}

    def walk(path: Path, rel: Path, depth: int) -> None:
        # Never follow symlinks (a dir symlink could loop or escape the tree),
        # and cap nesting depth (stops archive-quines and pathological trees).
        if path.is_symlink() or _is_junk(path.name):
            stats["skipped"] += 1
            return
        if depth > _MAX_DEPTH:
            logger.warning("max nesting depth reached, skipping: %s", rel)
            stats["skipped"] += 1
            return
        try:
            if path.is_dir():
                for child in sorted(path.iterdir()):
                    walk(child, rel / child.name, depth + 1)
                return
            if is_archive(path):
                stats["archives"] += 1
                with tempfile.TemporaryDirectory(prefix="keith_convert_") as tmp:
                    try:
                        _extract_archive(path, Path(tmp))
                    except Exception as exc:  # noqa: BLE001 - a bad archive must not stop the run
                        logger.warning("could not extract archive %s: %s", rel, exc)
                        return
                    for child in sorted(Path(tmp).iterdir()):
                        walk(child, rel / child.name, depth + 1)  # nest under the archive name
                return
            if path.suffix.lower() not in CONVERTIBLE_EXTS:
                stats["skipped"] += 1
                return
            # Append .txt to the full name (not replace the suffix) so foo.pdf and
            # foo.html don't both collapse to foo.txt and clobber each other.
            dest = out / rel.parent / f"{rel.name}.txt"
            if not force and _is_fresh(dest, path):
                stats["cached"] += 1
                return
            text = convert_file(path, enable_ocr, do_reflow, do_fix_spacing, min_chars)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
            logger.warning("conversion failed for %s: %s", rel, exc)
            stats["failed"] += 1
            return
        if text is None:
            stats["rejected"] += 1
            logger.info("discarded (unreadable / no text): %s", rel)
            return
        # Write atomically: a run killed mid-write must never leave a truncated
        # but non-empty .txt, which _is_fresh would then treat as complete and
        # cache forever. Write a temp sibling, then atomically rename over dest.
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, dest)
        except OSError as exc:  # noqa: BLE001 - a write failure must not stop the run
            logger.warning("could not write %s: %s", dest, exc)
            tmp.unlink(missing_ok=True)
            stats["failed"] += 1
            return
        stats["converted"] += 1

    for child in sorted(src.iterdir()):
        walk(child, Path(child.name), 0)
    logger.info("conversion complete: %s", stats)
    return stats
