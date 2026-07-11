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
import re
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from keith_llm.data.archive import _SINGLE_OPENERS, is_archive
from keith_llm.data.audit import score_text
from keith_llm.data.clean import clean_pages, clean_text, is_quality
from keith_llm.data.ocr import ocr_available
from keith_llm.data.pdf_layout import extract_pdf_pages

logger = logging.getLogger(__name__)

PDF_EXTS = {".pdf"}
HTML_EXTS = {".html", ".htm", ".xhtml"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
TEXT_EXTS = {".txt", ".md", ".markdown"}
CONVERTIBLE_EXTS = PDF_EXTS | HTML_EXTS | IMAGE_EXTS | TEXT_EXTS

_LONG_TOKEN = re.compile(r"[A-Za-z]{19,}")  # candidate clumped-word runs
_BLANKLINE = re.compile(r"\n\s*\n")


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
        if is_readable(ocr_text, min_chars) or len(ocr_text) > len(text):
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
    else:
        return None

    text = raw
    if do_reflow:
        text = reflow(text)
    if do_fix_spacing:
        text = fix_spacing(text)
    return text if is_readable(text, min_chars) else None


# --- archives ---


def _extract_archive(path: Path, dest: Path) -> None:
    """Safely extract an archive into ``dest`` (traversal-guarded)."""
    suffix = path.suffix.lower()
    suffixes = [s.lower() for s in path.suffixes]
    is_tar = suffix in {".tar", ".tgz", ".tbz2", ".tbz", ".txz"} or (
        len(suffixes) >= 2 and suffixes[-2] == ".tar"
    )
    if suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            zf.extractall(dest)  # zipfile sanitizes member paths
    elif is_tar:
        with tarfile.open(path) as tf:
            tf.extractall(dest, filter="data")  # rejects traversal/special files
    elif suffix in _SINGLE_OPENERS:
        inner = dest / path.name[: -len(path.suffix)]
        with _SINGLE_OPENERS[suffix](path, "rb") as src, inner.open("wb") as out:
            shutil.copyfileobj(src, out)


# --- tree walk ---


def convert_tree(
    src: str | Path,
    out: str | Path,
    enable_ocr: bool = True,
    do_reflow: bool = True,
    do_fix_spacing: bool = True,
    min_chars: int = 200,
) -> dict[str, Any]:
    """Convert every convertible file under ``src`` into readable ``.txt`` files
    mirrored under ``out``. Recurses subdirectories and archives; discards
    unreadable results. Returns summary stats."""
    src, out = Path(src), Path(out)
    stats = {"converted": 0, "rejected": 0, "skipped": 0, "archives": 0, "failed": 0}

    def walk(path: Path, rel: Path) -> None:
        try:
            if path.is_dir():
                for child in sorted(path.iterdir()):
                    walk(child, rel / child.name)
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
                        walk(child, rel / child.name)  # nest outputs under the archive name
                return
            if path.suffix.lower() not in CONVERTIBLE_EXTS:
                stats["skipped"] += 1
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
        dest = out / rel.with_suffix(".txt")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        stats["converted"] += 1

    src = src.resolve()
    for child in sorted(src.iterdir()):
        walk(child, Path(child.name))
    logger.info("conversion complete: %s", stats)
    return stats
