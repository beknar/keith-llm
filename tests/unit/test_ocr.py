from keith_llm.data import ocr, pdf_layout


def test_ocr_available_is_a_bool_and_does_not_crash():
    # Without the ocr extra / tesseract installed it must be False, never raise.
    assert isinstance(ocr.ocr_available(), bool)


def test_nonspace_len():
    assert pdf_layout._nonspace_len("  a b\tc\n") == 3
    assert pdf_layout._nonspace_len("   \n\t ") == 0


def test_ocr_fills_only_image_only_pages(monkeypatch):
    pages = ["Real page one with plenty of extractable prose text here.", "   \n  "]
    monkeypatch.setattr(ocr, "ocr_available", lambda: True)

    called = {}

    def fake_ocr(path, indices, **kw):
        called["indices"] = set(indices)
        return {i: "Recovered scanned text from the image page." for i in indices}

    monkeypatch.setattr(ocr, "ocr_pdf_pages", fake_ocr)
    out = pdf_layout._ocr_fill("scan.pdf", list(pages))
    assert called["indices"] == {1}  # only the near-empty page was OCR'd
    assert out[0] == pages[0]  # text page untouched
    assert "Recovered scanned text" in out[1]


def test_ocr_skipped_when_unavailable(monkeypatch):
    monkeypatch.setattr(ocr, "ocr_available", lambda: False)

    def boom(*a, **k):
        raise AssertionError("ocr_pdf_pages must not be called when OCR is unavailable")

    monkeypatch.setattr(ocr, "ocr_pdf_pages", boom)
    pages = ["", "   "]
    assert pdf_layout._ocr_fill("scan.pdf", list(pages)) == pages


def test_ocr_not_run_when_all_pages_have_text(monkeypatch):
    monkeypatch.setattr(ocr, "ocr_available", lambda: True)

    def boom(*a, **k):
        raise AssertionError("no page needs OCR")

    monkeypatch.setattr(ocr, "ocr_pdf_pages", boom)
    pages = ["Full page of text one here.", "Full page of text two here."]
    assert pdf_layout._ocr_fill("doc.pdf", list(pages)) == pages


def test_ocr_failure_degrades_gracefully(monkeypatch):
    monkeypatch.setattr(ocr, "ocr_available", lambda: True)
    monkeypatch.setattr(
        ocr, "ocr_pdf_pages", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("tesseract died"))
    )
    pages = ["good text here on page one", "  "]
    # OCR blows up -> original (empty) page kept, no exception propagates
    assert pdf_layout._ocr_fill("scan.pdf", list(pages)) == pages


def test_ocr_keeps_better_of_original_and_ocr(monkeypatch):
    # If OCR returns less text than what was already there, keep the original.
    monkeypatch.setattr(ocr, "ocr_available", lambda: True)
    monkeypatch.setattr(ocr, "ocr_pdf_pages", lambda path, idx, **k: {0: "x"})
    pages = [""]  # empty -> needs OCR; OCR gives "x" (1 char) which beats 0
    assert pdf_layout._ocr_fill("scan.pdf", list(pages)) == ["x"]


def test_extract_pdf_pages_end_to_end_ocr(monkeypatch, make_pdf, tmp_path):
    # A real (text) PDF whose page pdfplumber reads as empty gets OCR-filled.
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(make_pdf(["placeholder"]))
    monkeypatch.setattr(pdf_layout, "_pdfplumber_pages", lambda p: [""])
    monkeypatch.setattr(pdf_layout, "_pypdf_pages", lambda p: [""])
    monkeypatch.setattr(ocr, "ocr_available", lambda: True)
    monkeypatch.setattr(ocr, "ocr_pdf_pages", lambda p, idx, **k: {0: "scanned content recovered"})
    assert pdf_layout.extract_pdf_pages(str(pdf)) == ["scanned content recovered"]


def test_extract_pdf_pages_ocr_disabled(monkeypatch, make_pdf, tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(make_pdf(["placeholder"]))
    monkeypatch.setattr(pdf_layout, "_pdfplumber_pages", lambda p: [""])
    monkeypatch.setattr(pdf_layout, "_pypdf_pages", lambda p: [""])
    monkeypatch.setattr(ocr, "ocr_available", lambda: True)
    monkeypatch.setattr(
        ocr, "ocr_pdf_pages", lambda *a, **k: (_ for _ in ()).throw(AssertionError("disabled"))
    )
    assert pdf_layout.extract_pdf_pages(str(pdf), enable_ocr=False) == [""]
