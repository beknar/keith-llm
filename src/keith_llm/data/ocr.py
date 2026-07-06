"""Optical character recognition for image-only (scanned) PDF pages.

Scanned PDFs have no text layer, so pdfplumber/pypdf recover nothing and the
document is dropped by the quality filter. This module rasterizes such pages
and runs Tesseract on them to recover the text.

OCR is an OPTIONAL capability: it needs the ``ocr`` extra (``pypdfium2`` for
rendering, ``pytesseract`` for recognition) plus the system ``tesseract``
binary. If any of those is missing, :func:`ocr_available` returns False and the
pipeline behaves exactly as before (scans are dropped) — nothing breaks, OCR
just doesn't happen. Install it with::

    pip install -e ".[ocr]"        # plus: apt-get install tesseract-ocr
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_DEFAULT_DPI = 300
_DEFAULT_LANG = "eng"


def ocr_available() -> bool:
    """True only if rendering + OCR can actually run (deps and the tesseract
    binary all present)."""
    try:
        import pypdfium2  # noqa: F401
        import pytesseract
    except ImportError:
        return False
    try:
        pytesseract.get_tesseract_version()
    except Exception:  # noqa: BLE001 - binary missing or unrunnable
        return False
    return True


def ocr_pdf_pages(
    path: str,
    page_indices,
    dpi: int = _DEFAULT_DPI,
    lang: str = _DEFAULT_LANG,
) -> dict[int, str]:
    """Render the requested page indices at ``dpi`` and OCR them.

    Returns ``{page_index: recognized_text}``. A page that fails to render or
    OCR is logged and omitted rather than aborting the whole document.
    """
    import pypdfium2 as pdfium
    import pytesseract

    scale = dpi / 72.0
    out: dict[int, str] = {}
    pdf = pdfium.PdfDocument(path)
    try:
        n_pages = len(pdf)
        for i in sorted(set(page_indices)):
            if not (0 <= i < n_pages):
                continue
            try:
                page = pdf[i]
                image = page.render(scale=scale).to_pil()
                out[i] = pytesseract.image_to_string(image, lang=lang)
            except Exception as exc:  # noqa: BLE001 - one bad page must not kill the doc
                logger.warning("OCR failed on page %d of %s: %s", i, path, exc)
    finally:
        pdf.close()
    return out
