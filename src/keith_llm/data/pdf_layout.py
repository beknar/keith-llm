"""Geometry-aware PDF text extraction.

``pypdf`` reads text in raw content-stream order, which on the multi-column
layouts typical of rulebooks and adventure modules interleaves the columns or
reads them out of order — silently corrupting sentences. This module extracts
word bounding boxes with ``pdfplumber``, detects columns from the horizontal
whitespace between them, and reads each column top-to-bottom so the output
follows human reading order.

The ordering logic (:func:`order_page_words`) is a pure function over word
boxes so it can be tested without any PDF. :func:`extract_pdf_pages` is the
thin I/O adapter, and it falls back to pypdf if pdfplumber is unavailable or
errors on a file.

Known limitation: a full-width heading that straddles the gutter is assigned
to whichever column its midpoint lands in, so it may attach to one column
rather than sitting above both. Body text ordering — the thing that actually
corrupts training data — is unaffected.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)


class Word(TypedDict):
    """A positioned word. Matches the keys pdfplumber's extract_words emits;
    ``top``/``bottom`` are distances from the page top (smaller = higher)."""

    text: str
    x0: float
    x1: float
    top: float
    bottom: float


# Below this many words a page is not worth column analysis (title/art pages);
# geometric top-to-bottom reading is already correct and splitting risks harm.
_MIN_WORDS_FOR_COLUMNS = 25
# A detected column count above this usually means a table's internal gaps were
# mistaken for gutters; collapse back to a single column instead of shredding.
_MAX_COLUMNS = 4


def _detect_columns(
    words: list[Word], xmin: float, xmax: float, min_gutter: float
) -> list[tuple[float, float]]:
    """Return column x-bands, left to right, by finding vertical whitespace.

    Each word contributes to the horizontal bins its span covers. Body columns
    accumulate many overlaps; a gutter — even one crossed by the occasional
    full-width heading — stays far below the column peak, so thresholding the
    per-bin overlap count against a fraction of the peak isolates the columns.
    """
    width = xmax - xmin
    if width <= 0:
        return [(xmin, xmax)]
    n_bins = max(50, int(width))
    bin_w = width / n_bins
    counts = [0] * n_bins
    for w in words:
        lo = max(0, int((w["x0"] - xmin) / bin_w))
        hi = min(n_bins - 1, int((w["x1"] - xmin) / bin_w))
        for b in range(lo, hi + 1):
            counts[b] += 1

    peak = max(counts)
    threshold = max(1.0, 0.10 * peak)  # bins at/below this are gutter/margin

    columns: list[list[float]] = []
    b = 0
    while b < n_bins:
        if counts[b] > threshold:
            start = b
            while b < n_bins and counts[b] > threshold:
                b += 1
            columns.append([xmin + start * bin_w, xmin + b * bin_w])
        else:
            b += 1

    if not columns:
        return [(xmin, xmax)]

    # Merge bands separated by a sub-gutter-width gap (intra-paragraph spacing).
    merged: list[list[float]] = [columns[0]]
    for lo, hi in columns[1:]:
        if lo - merged[-1][1] < min_gutter:
            merged[-1][1] = hi
        else:
            merged.append([lo, hi])

    if len(merged) > _MAX_COLUMNS:
        return [(xmin, xmax)]
    return [(lo, hi) for lo, hi in merged]


def _group_lines(col_words: list[Word]) -> str:
    """Order one column's words into lines, top-to-bottom, left-to-right."""
    col_words = sorted(col_words, key=lambda w: (w["top"], w["x0"]))
    heights = sorted(max(w["bottom"] - w["top"], 0.0) for w in col_words)
    median_h = heights[len(heights) // 2] or 1.0
    tol = 0.6 * median_h

    lines: list[list[Word]] = []
    line_top = col_words[0]["top"]
    current: list[Word] = []
    for w in col_words:
        if current and abs(w["top"] - line_top) > tol:
            lines.append(current)
            current = []
            line_top = w["top"]
        current.append(w)
        if len(current) == 1:
            line_top = w["top"]
    if current:
        lines.append(current)

    return "\n".join(
        " ".join(w["text"] for w in sorted(line, key=lambda w: w["x0"])) for line in lines
    )


def order_page_words(words: list[Word], page_width: float | None = None) -> str:
    """Render one page's positioned words into reading-order plain text."""
    words = [w for w in words if w["text"].strip()]
    if not words:
        return ""
    xmin = min(w["x0"] for w in words)
    xmax = max(w["x1"] for w in words)
    if page_width is None:
        page_width = xmax

    if len(words) < _MIN_WORDS_FOR_COLUMNS:
        columns: list[tuple[float, float]] = [(xmin, xmax)]
    else:
        min_gutter = max(6.0, 0.02 * page_width)
        columns = _detect_columns(words, xmin, xmax, min_gutter)

    if len(columns) == 1:
        return _group_lines(words)

    midpoints = [(lo + hi) / 2 for lo, hi in columns]
    buckets: list[list[Word]] = [[] for _ in columns]
    for w in words:
        mid = (w["x0"] + w["x1"]) / 2
        assigned = next((i for i, (lo, hi) in enumerate(columns) if lo <= mid <= hi), None)
        if assigned is None:
            assigned = min(range(len(columns)), key=lambda i: abs(midpoints[i] - mid))
        buckets[assigned].append(w)

    return "\n".join(_group_lines(bucket) for bucket in buckets if bucket)


def _pypdf_pages(path: str) -> list[str]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    return [(page.extract_text() or "") for page in reader.pages]


def _pdfplumber_pages(path: str) -> list[str]:
    import pdfplumber

    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
            pages.append(order_page_words(words, page_width=page.width))
    return pages


# A page with fewer than this many non-space characters of extractable text is
# treated as image-only (a scan) and sent to OCR when it is available.
_OCR_MIN_CHARS = 32


def _nonspace_len(text: str) -> int:
    return sum(1 for c in text if not c.isspace())


def _ocr_fill(path: str, pages: list[str]) -> list[str]:
    """Replace image-only pages with OCR'd text, if OCR is available."""
    from keith_llm.data.ocr import ocr_available, ocr_pdf_pages

    if not ocr_available():
        return pages
    needed = {i for i, p in enumerate(pages) if _nonspace_len(p) < _OCR_MIN_CHARS}
    if not needed:
        return pages
    logger.info(
        "OCR: %d/%d pages of %s look image-only; running Tesseract",
        len(needed),
        len(pages),
        Path(path).name,
    )
    try:
        recovered = ocr_pdf_pages(path, needed)
    except Exception as exc:  # noqa: BLE001 - OCR failure must not kill ingestion
        logger.warning("OCR failed for %s: %s", path, exc)
        return pages
    for i, text in recovered.items():
        if _nonspace_len(text) > _nonspace_len(pages[i]):
            pages[i] = text
    return pages


def extract_pdf_pages(path: str, enable_ocr: bool = True) -> list[str]:
    """Extract a PDF as one string per page, column-aware where possible.

    Uses pdfplumber for geometry-aware ordering, falling back to pypdf if
    pdfplumber is missing or errors. Image-only pages (scans) are then recovered
    with OCR when it is available (``enable_ocr`` and the ``ocr`` extra); without
    OCR they stay empty and the document is dropped downstream as before.
    """
    try:
        import pdfplumber  # noqa: F401

        pages = _pdfplumber_pages(path)
    except ImportError:
        logger.info("pdfplumber not installed; using pypdf for %s", path)
        pages = _pypdf_pages(path)
    except Exception as exc:  # noqa: BLE001 - any parse failure should degrade, not crash
        logger.warning("pdfplumber failed on %s (%s); falling back to pypdf", path, exc)
        pages = _pypdf_pages(path)

    # If geometry extraction yielded nothing, a pypdf retry occasionally helps.
    if not any(p.strip() for p in pages):
        alt = _pypdf_pages(path)
        if any(p.strip() for p in alt):
            pages = alt

    if enable_ocr:
        pages = _ocr_fill(path, pages)
    return pages
