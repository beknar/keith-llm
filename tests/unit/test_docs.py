import zipfile

from keith_llm.data.convert import convert_file, convert_tree
from keith_llm.data.docs import DOC_EXTS, extract_document

PROSE = (
    "The village of Emberfall sits at the edge of the mirewood, its palisade "
    "scarred by last winter raids. The elders speak of a barrow beneath the old "
    "mill where something ancient turns in its sleep, and they will pay good "
    "silver to anyone brave enough to see that it stays sleeping forever more."
)

_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


# --- minimal fixture builders (stdlib only, so these run in CI) ---


def _make_docx(path, paragraphs):
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    doc = f'<?xml version="1.0"?><w:document xmlns:w="{_W}"><w:body>{body}</w:body></w:document>'
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", doc)


def _make_odt(path, paragraphs):
    t = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
    o = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"
    body = "".join(f"<text:p>{p}</text:p>" for p in paragraphs)
    content = (
        f'<?xml version="1.0"?><office:document-content xmlns:office="{o}" '
        f'xmlns:text="{t}"><office:body><office:text>{body}'
        "</office:text></office:body></office:document-content>"
    )
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("content.xml", content)


def _make_epub(path, chapters):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:'
            'xmlns:container"><rootfiles><rootfile full-path="content.opf"/>'
            "</rootfiles></container>",
        )
        items = "".join(f'<item id="c{i}" href="ch{i}.xhtml"/>' for i in range(len(chapters)))
        refs = "".join(f'<itemref idref="c{i}"/>' for i in range(len(chapters)))
        z.writestr(
            "content.opf",
            f'<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf">'
            f"<manifest>{items}</manifest><spine>{refs}</spine></package>",
        )
        for i, ch in enumerate(chapters):
            z.writestr(f"ch{i}.xhtml", f"<html><body><p>{ch}</p></body></html>")


# --- extractors ---


def test_docx_extracts_paragraphs(tmp_path):
    p = tmp_path / "a.docx"
    _make_docx(p, ["First paragraph about goblins.", "Second paragraph about dragons."])
    text = extract_document(p)
    assert "First paragraph about goblins." in text
    assert "Second paragraph about dragons." in text
    assert "\n" in text  # paragraphs separated


def test_odt_extracts_paragraphs(tmp_path):
    p = tmp_path / "a.odt"
    _make_odt(p, ["Chapter one text.", "Chapter two text."])
    text = extract_document(p)
    assert "Chapter one text." in text
    assert "Chapter two text." in text


def test_epub_extracts_chapters_in_spine_order(tmp_path):
    p = tmp_path / "a.epub"
    _make_epub(p, ["Alpha chapter content.", "Beta chapter content."])
    text = extract_document(p)
    assert "Alpha chapter content." in text
    assert "Beta chapter content." in text
    assert text.index("Alpha") < text.index("Beta")  # spine order preserved


def test_docx_strict_ooxml_namespace(tmp_path):
    # "Strict Open XML" uses a different namespace; local-name matching must still
    # extract text (the transitional namespace must not be hardcoded)
    strict = "http://purl.oclc.org/ooxml/wordprocessingml/main"
    p = tmp_path / "strict.docx"
    body = "<w:p><w:r><w:t>Strict namespace prose here.</w:t></w:r></w:p>"
    doc = (
        f'<?xml version="1.0"?><w:document xmlns:w="{strict}"><w:body>{body}</w:body></w:document>'
    )
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("word/document.xml", doc)
    assert "Strict namespace prose here." in extract_document(p)


def test_docx_separates_paragraphs_with_blank_line(tmp_path):
    # paragraphs must survive reflow -> blank-line separated, not merged to one line
    p = tmp_path / "multi.docx"
    _make_docx(p, ["Para one.", "Para two."])
    assert extract_document(p) == "Para one.\n\nPara two."


def _make_epub_custom(path, opf_path, chapter_member, chapter_href):
    """Build an epub with a chosen OPF location and a spine href to test path resolution."""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:'
            'xmlns:container"><rootfiles>'
            f'<rootfile full-path="{opf_path}"/></rootfiles></container>',
        )
        z.writestr(
            opf_path,
            '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf"><manifest>'
            f'<item id="c0" href="{chapter_href}"/></manifest><spine><itemref idref="c0"/>'
            "</spine></package>",
        )
        z.writestr(chapter_member, "<html><body><p>Resolved chapter text.</p></body></html>")


def test_epub_resolves_parent_relative_href(tmp_path):
    # OPF in OPS/, chapter in Text/, href uses ../ -> must resolve to Text/ch.xhtml
    p = tmp_path / "rel.epub"
    _make_epub_custom(p, "OPS/package.opf", "Text/ch.xhtml", "../Text/ch.xhtml")
    assert "Resolved chapter text." in extract_document(p)


def test_epub_resolves_percent_encoded_href(tmp_path):
    p = tmp_path / "pct.epub"
    _make_epub_custom(p, "content.opf", "Chapter 1.xhtml", "Chapter%201.xhtml")
    assert "Resolved chapter text." in extract_document(p)


def test_epub_falls_back_to_sorted_when_no_opf(tmp_path):
    # an epub with html files but a broken/missing OPF still yields text
    p = tmp_path / "b.epub"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("chap1.xhtml", "<html><body><p>Fallback chapter body.</p></body></html>")
    assert "Fallback chapter body." in extract_document(p)


def test_rtf_extracts_text(tmp_path):
    try:
        import striprtf  # noqa: F401
    except ImportError:
        return  # optional dep absent -> skip cleanly
    p = tmp_path / "a.rtf"
    p.write_text(r"{\rtf1\ansi The goblin waits in the dark cave.\par Second line here.}")
    text = extract_document(p)
    assert "goblin waits in the dark cave" in text


# --- dispatch / graceful failure ---


def test_unsupported_extension_returns_none(tmp_path):
    p = tmp_path / "a.xyz"
    p.write_text("whatever")
    assert extract_document(p) is None


def test_corrupt_docx_returns_none(tmp_path):
    p = tmp_path / "bad.docx"
    p.write_bytes(b"not a zip at all")
    assert extract_document(p) is None  # exception swallowed -> None, run continues


def test_empty_extraction_returns_none(tmp_path):
    p = tmp_path / "empty.docx"
    _make_docx(p, ["   ", ""])  # no real text
    assert extract_document(p) is None


def test_doc_without_tool_returns_none(tmp_path, monkeypatch):
    import shutil

    # no antiword/catdoc available -> legacy .doc is unconvertible, not a crash
    monkeypatch.setattr(shutil, "which", lambda _: None)
    p = tmp_path / "legacy.doc"
    p.write_bytes(b"\xd0\xcf\x11\xe0 old OLE doc bytes")
    assert extract_document(p) is None


def test_mobi_without_lib_returns_none(tmp_path, monkeypatch):
    # simulate the mobi package being absent -> graceful None, no crash
    import builtins

    real_import = builtins.__import__

    def no_mobi(name, *a, **k):
        if name == "mobi":
            raise ImportError("no mobi")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_mobi)
    p = tmp_path / "book.mobi"
    p.write_bytes(b"BOOKMOBI fake")
    assert extract_document(p) is None


def test_doc_exts_are_registered():
    assert {".docx", ".doc", ".odt", ".rtf", ".epub", ".mobi"} == set(DOC_EXTS)


# --- integration through convert_file / convert_tree ---


def test_convert_file_handles_docx(tmp_path):
    p = tmp_path / "story.docx"
    _make_docx(p, [PROSE, "The party ventures onward into the mire. " * 5])
    out = convert_file(p, enable_ocr=False, do_fix_spacing=False, min_chars=50)
    assert out is not None
    assert "Emberfall" in out


def test_convert_tree_picks_up_epub(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _make_epub(src / "book.epub", [PROSE, "A long second chapter. " * 20])
    out = tmp_path / "out"
    stats = convert_tree(src, out, enable_ocr=False, do_fix_spacing=False, min_chars=50)
    assert stats["converted"] == 1
    assert (out / "book.epub.txt").exists()
    assert "Emberfall" in (out / "book.epub.txt").read_text()
