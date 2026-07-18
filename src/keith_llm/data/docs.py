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
import posixpath
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from urllib.parse import unquote

logger = logging.getLogger(__name__)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _docx(path: str) -> str:
    """Extract body text from a .docx (OOXML): a paragraph per ``<w:p>``. Matches
    by local tag name so both transitional and Strict Open XML namespaces work."""
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    paras = []
    for el in root.iter():
        if _local(el.tag) == "p":
            text = "".join(t.text or "" for t in el.iter() if _local(t.tag) == "t")
            if text.strip():
                paras.append(text)
    return "\n\n".join(paras)


def _odt(path: str) -> str:
    """Extract text from an .odt (OpenDocument): a paragraph per heading/para."""
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("content.xml"))
    paras = []
    for el in root.iter():
        if _local(el.tag) in ("p", "h"):
            text = "".join(el.itertext())
            if text.strip():
                paras.append(text)
    return "\n\n".join(paras)


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
        # Resolve each href against the OPF's directory: percent-decode, join,
        # and normalize so subdir OPFs and ../-relative hrefs map to real members.
        ordered = []
        for s in spine:
            href = manifest.get(s)
            if href:
                ordered.append(posixpath.normpath(posixpath.join(base, unquote(href))))
        if ordered:
            return ordered
    except Exception:  # noqa: BLE001 - malformed OPF -> fall back to sorted scan
        pass
    return _sorted_html_members(z)


def _sorted_html_members(z: zipfile.ZipFile) -> list[str]:
    return sorted(n for n in z.namelist() if n.lower().endswith((".xhtml", ".html", ".htm")))


def _read_html_members(z: zipfile.ZipFile, names, html_to_text) -> list[str]:
    parts = []
    for name in names:
        try:
            html = z.read(name).decode("utf-8", "replace")
        except KeyError:
            continue  # spine listed a member that isn't in the zip
        text = html_to_text(html)
        if text.strip():
            parts.append(text)
    return parts


def _epub(path: str) -> str:
    """Extract text from an .epub by stripping each spine chapter's XHTML. If the
    spine hrefs don't resolve to any readable members, fall back to every html
    file in the archive so a quirky OPF can't lose the whole book."""
    from keith_llm.data.convert import html_to_text

    with zipfile.ZipFile(path) as z:
        parts = _read_html_members(z, _epub_html_members(z), html_to_text)
        if not parts:
            parts = _read_html_members(z, _sorted_html_members(z), html_to_text)
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

    try:
        import mobi
    except ImportError:
        logger.info("mobi package not installed; cannot convert .mobi: %s", path)
        return None

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
