"""Text extraction for word-processor and ebook formats.

The convert pipeline handles PDF/HTML/image/text; this recovers the *other*
digital document formats that would otherwise be skipped — Word (.docx/.doc),
OpenDocument (.odt), RTF, and ebooks (.epub/.mobi). Unlike scanned PDFs these
are already-digital text, so extraction is clean and needs no OCR.

Each extractor lazily imports what it needs and any failure returns ``None``
(the caller then treats the file as unconvertible), so a missing optional
library or a malformed file never aborts a run:

- ``.docx``/``.odt``/``.epub`` — pure stdlib (zipfile + XML), reusing the
  convert HTML stripper for epub chapters. Always available.
- ``.rtf`` — needs ``striprtf`` (part of the ``convert`` extra).
- ``.mobi`` — needs the ``mobi`` package (part of the ``convert`` extra).
- ``.doc`` (legacy binary Word) — needs a system tool (``antiword`` or
  ``catdoc``); returns ``None`` if neither is installed.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _docx(path: str) -> str:
    """Extract body text from a .docx (OOXML): one line per ``<w:p>``."""
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    paras = []
    for p in root.iter(f"{_W}p"):
        text = "".join(t.text or "" for t in p.iter(f"{_W}t"))
        if text.strip():
            paras.append(text)
    return "\n".join(paras)


def _odt(path: str) -> str:
    """Extract text from an .odt (OpenDocument): one line per paragraph/heading."""
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("content.xml"))
    paras = []
    for el in root.iter():
        if _local(el.tag) in ("p", "h"):
            text = "".join(el.itertext())
            if text.strip():
                paras.append(text)
    return "\n".join(paras)


def _epub_html_members(z: zipfile.ZipFile) -> list[str]:
    """Ordered list of the epub's (x)html content files. Follows the OPF spine
    for reading order; falls back to all html-ish members sorted by name."""
    try:
        container = ET.fromstring(z.read("META-INF/container.xml"))
        opf_path = next(
            rf.get("full-path")
            for rf in container.iter()
            if _local(rf.tag) == "rootfile" and rf.get("full-path")
        )
        opf = ET.fromstring(z.read(opf_path))
        base = opf_path.rsplit("/", 1)[0] if "/" in opf_path else ""
        manifest, spine = {}, []
        for el in opf.iter():
            tag = _local(el.tag)
            if tag == "item":
                manifest[el.get("id")] = el.get("href")
            elif tag == "itemref":
                spine.append(el.get("idref"))
        ordered = [
            f"{base}/{manifest[s]}" if base else manifest[s] for s in spine if manifest.get(s)
        ]
        if ordered:
            return ordered
    except Exception:  # noqa: BLE001 - malformed OPF -> fall back to sorted scan
        pass
    return sorted(n for n in z.namelist() if n.lower().endswith((".xhtml", ".html", ".htm")))


def _epub(path: str) -> str:
    """Extract text from an .epub by stripping each spine chapter's XHTML."""
    from keith_llm.data.convert import html_to_text

    parts = []
    with zipfile.ZipFile(path) as z:
        for name in _epub_html_members(z):
            try:
                html = z.read(name).decode("utf-8", "replace")
            except KeyError:
                continue
            text = html_to_text(html)
            if text.strip():
                parts.append(text)
    return "\n\n".join(parts)


def _rtf(path: str) -> str:
    from striprtf.striprtf import rtf_to_text

    with open(path, encoding="utf-8", errors="replace") as fh:
        return rtf_to_text(fh.read())


def _doc(path: str) -> str | None:
    """Legacy binary .doc via a system tool (antiword/catdoc) if present."""
    import shutil
    import subprocess

    exe = shutil.which("antiword") or shutil.which("catdoc")
    if not exe:
        logger.info("no antiword/catdoc; cannot convert legacy .doc: %s", path)
        return None
    proc = subprocess.run([exe, path], capture_output=True, timeout=180)
    return proc.stdout.decode("utf-8", "replace") if proc.returncode == 0 else None


def _mobi(path: str) -> str | None:
    import shutil

    import mobi

    from keith_llm.data.convert import html_to_text

    tmpdir, filepath = mobi.extract(path)
    try:
        if filepath.lower().endswith(".epub"):
            return _epub(filepath)
        with open(filepath, encoding="utf-8", errors="replace") as fh:
            return html_to_text(fh.read())
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


_DISPATCH = {
    ".docx": _docx,
    ".odt": _odt,
    ".epub": _epub,
    ".rtf": _rtf,
    ".doc": _doc,
    ".mobi": _mobi,
}

DOC_EXTS = frozenset(_DISPATCH)


def extract_document(path: str | Path) -> str | None:
    """Extract plain text from a supported document/ebook file, or ``None`` if
    the format is unsupported, its library/tool is missing, or extraction fails
    or yields nothing usable."""
    fn = _DISPATCH.get(Path(path).suffix.lower())
    if fn is None:
        return None
    try:
        text = fn(str(path))
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
        logger.warning("document extraction failed for %s: %s", path, exc)
        return None
    return text if text and text.strip() else None
