import pytest

from keith_llm.data.ingest import extract_pages


def test_txt_extraction(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("The goblin ambush begins at dusk.")
    assert extract_pages(f) == ["The goblin ambush begins at dusk."]


def test_md_extraction(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# Emberfall\n\nA village on the edge of the mirewood.")
    pages = extract_pages(f)
    assert len(pages) == 1
    assert "mirewood" in pages[0]


def test_pdf_extraction(tmp_path, make_pdf):
    f = tmp_path / "doc.pdf"
    f.write_bytes(make_pdf(["The goblin ambush begins at dusk.", "Roll for initiative."]))
    pages = extract_pages(f)
    assert len(pages) == 2
    assert "goblin ambush" in pages[0]
    assert "initiative" in pages[1]


def test_unsupported_extension(tmp_path):
    f = tmp_path / "doc.docx"
    f.write_text("nope")
    with pytest.raises(ValueError, match="unsupported"):
        extract_pages(f)
