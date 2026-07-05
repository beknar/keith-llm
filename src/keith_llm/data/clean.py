"""Text cleanup: header/footer stripping, encoding repair, quality filtering."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

import ftfy

_HYPHEN_WRAP = re.compile(r"(\w)-\n(\w)")
_TRAILING_WS = re.compile(r"[ \t]+\n")
_MULTI_BLANK = re.compile(r"\n{3,}")
_DIGITS = re.compile(r"\d+")


def _line_key(line: str) -> str:
    # Digits are collapsed so "Page 3" and "Page 71" count as the same line.
    return _DIGITS.sub("#", line.strip()).lower()


def clean_pages(pages: list[str], min_pages: int = 4, max_line_frac: float = 0.5) -> str:
    """Join pages, dropping lines that repeat on more than ``max_line_frac``
    of pages (running headers, footers, page numbers).

    Documents with fewer than ``min_pages`` pages are joined untouched — the
    frequency signal is meaningless there.
    """
    if len(pages) < min_pages:
        return "\n".join(pages)
    counts: Counter[str] = Counter()
    for page in pages:
        counts.update({_line_key(ln) for ln in page.splitlines() if ln.strip()})
    cutoff = max_line_frac * len(pages)
    boiler = {key for key, n in counts.items() if n > cutoff}
    kept_pages = []
    for page in pages:
        kept = [ln for ln in page.splitlines() if not ln.strip() or _line_key(ln) not in boiler]
        kept_pages.append("\n".join(kept))
    return "\n".join(kept_pages)


def clean_text(text: str) -> str:
    text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_WRAP.sub(r"\1\2", text)
    text = _TRAILING_WS.sub("\n", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


def is_quality(text: str, min_chars: int = 200, min_alpha_frac: float = 0.5) -> bool:
    """Lenient filter: TTRPG stat blocks are table-heavy, so the alpha-ratio
    threshold is deliberately low."""
    if len(text) < min_chars:
        return False
    non_ws = [c for c in text if not c.isspace()]
    if not non_ws:
        return False
    alpha = sum(c.isalpha() for c in non_ws)
    return alpha / len(non_ws) > min_alpha_frac
