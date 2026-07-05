"""File-level text extraction.

PDFs return one string per page so the cleaner can strip running
headers/footers by cross-page frequency; plain-text formats return a single
"page".
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

TEXT_EXTS = {".txt", ".md", ".markdown"}
SUPPORTED_EXTS = TEXT_EXTS | {".pdf"}


def extract_pages(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return [(page.extract_text() or "") for page in reader.pages]
    if suffix in TEXT_EXTS:
        return [path.read_text(encoding="utf-8", errors="replace")]
    raise ValueError(f"unsupported file type: {path}")
