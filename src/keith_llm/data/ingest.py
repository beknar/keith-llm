"""File-level text extraction.

PDFs return one string per page so the cleaner can strip running
headers/footers by cross-page frequency; plain-text formats return a single
"page". PDF pages are read column-aware (see :mod:`keith_llm.data.pdf_layout`),
which fixes the multi-column reading-order corruption that plain pypdf causes
on rulebooks and modules.
"""

from __future__ import annotations

from pathlib import Path

from keith_llm.data.pdf_layout import extract_pdf_pages

TEXT_EXTS = {".txt", ".md", ".markdown"}
SUPPORTED_EXTS = TEXT_EXTS | {".pdf"}


def extract_pages(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_pages(str(path))
    if suffix in TEXT_EXTS:
        return [path.read_text(encoding="utf-8", errors="replace")]
    raise ValueError(f"unsupported file type: {path}")
