from keith_llm.data.pdf_layout import extract_pdf_pages, order_page_words


def w(text: str, x0: float, top: float, width: float = 40.0, height: float = 10.0):
    return {"text": text, "x0": x0, "x1": x0 + width, "top": top, "bottom": top + height}


# --- single column ---


def test_empty_page():
    assert order_page_words([]) == ""


def test_blank_words_skipped():
    assert order_page_words([w("   ", 50, 100)]) == ""


def test_single_word():
    assert order_page_words([w("Emberfall", 50, 100)]) == "Emberfall"


def test_single_column_reads_top_to_bottom():
    words = [w(f"line{i}", 50, 100 + i * 12) for i in range(30)]
    text = order_page_words(words, page_width=600)
    assert text.splitlines() == [f"line{i}" for i in range(30)]


def test_single_column_orders_words_within_a_line():
    line = [w("The", 50, 100), w("goblin", 90, 100), w("waits", 150, 100)]
    # supplied out of order; must sort by x0
    assert order_page_words(list(reversed(line)), page_width=600) == "The goblin waits"


# --- two column: the core reading-order fix ---


def _two_column_words(n_lines=30):
    words = []
    for i in range(n_lines):
        top = 100 + i * 12
        words.append(w(f"L{i}", 60, top))  # left column, x ~ 60-100
        words.append(w(f"R{i}", 340, top))  # right column, x ~ 340-380
    return words


def test_two_columns_not_interleaved():
    text = order_page_words(_two_column_words(), page_width=600)
    lines = text.splitlines()
    left = [ln for ln in lines if ln.startswith("L")]
    right = [ln for ln in lines if ln.startswith("R")]
    assert left == [f"L{i}" for i in range(30)]
    assert right == [f"R{i}" for i in range(30)]
    # every left line precedes every right line (columns read in full, in order)
    assert lines.index("L29") < lines.index("R0")


def test_two_columns_regression_vs_naive_reader():
    # The classic pypdf failure: same-row words across both columns. A naive
    # top-then-x reader yields "L0 R0 L1 R1..."; the column reader must not.
    text = order_page_words(_two_column_words(), page_width=600)
    assert "L0\nL1" in text
    assert "L0 R0" not in text


def test_full_width_heading_does_not_break_columns():
    words = _two_column_words()
    # a heading spanning both columns at the very top
    words.append(w("CHAPTER ONE", 60, 40, width=300))
    text = order_page_words(words, page_width=600)
    lines = text.splitlines()
    assert "CHAPTER ONE" in text
    # body columns still ordered and not interleaved
    assert [ln for ln in lines if ln.startswith("L")] == [f"L{i}" for i in range(30)]
    assert lines.index("L29") < lines.index("R0")


def test_three_columns():
    words = []
    for i in range(20):
        top = 100 + i * 12
        words.append(w(f"A{i}", 40, top))
        words.append(w(f"B{i}", 240, top))
        words.append(w(f"C{i}", 440, top))
    text = order_page_words(words, page_width=600)
    lines = text.splitlines()
    assert lines.index("A19") < lines.index("B0") < lines.index("B19") < lines.index("C0")


def test_sparse_page_stays_single_column():
    # Under the word threshold: do not attempt a split (avoids false columns).
    words = [w("Title", 60, 40), w("Subtitle", 340, 60), w("Author", 60, 400)]
    text = order_page_words(words, page_width=600)
    assert set(text.split()) == {"Title", "Subtitle", "Author"}


def test_wide_table_not_shredded_into_many_columns():
    # A 6-"column" grid of numbers (a table) should collapse to single column
    # rather than being read column-by-column.
    words = []
    for row in range(30):
        top = 100 + row * 12
        for c in range(6):
            words.append(w(str(row * 6 + c), 40 + c * 90, top, width=20))
    text = order_page_words(words, page_width=600)
    lines = text.splitlines()
    # single-column reading keeps each row intact on one line
    assert lines[0].split() == ["0", "1", "2", "3", "4", "5"]


# --- I/O adapter + fallback ---


def test_extract_pdf_pages_reads_real_pdf(tmp_path, make_pdf):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(make_pdf(["The goblin ambush begins at dusk.", "Roll for initiative."]))
    pages = extract_pdf_pages(str(pdf))
    assert len(pages) == 2
    assert "goblin ambush" in pages[0]
    assert "initiative" in pages[1]


def test_extract_pdf_pages_falls_back_when_pdfplumber_missing(tmp_path, make_pdf, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pdfplumber":
            raise ImportError("simulated missing pdfplumber")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(make_pdf(["Fallback path works."]))
    pages = extract_pdf_pages(str(pdf))
    assert "Fallback path works." in pages[0]


def test_extract_pdf_pages_falls_back_on_parse_error(tmp_path, monkeypatch):
    # pdfplumber raises -> pypdf path is used; here pypdf also fails cleanly,
    # so assert we at least reached the fallback rather than crashing in layout.
    from keith_llm.data import pdf_layout

    monkeypatch.setattr(
        pdf_layout, "_pdfplumber_pages", lambda p: (_ for _ in ()).throw(ValueError("boom"))
    )
    called = {}
    monkeypatch.setattr(pdf_layout, "_pypdf_pages", lambda p: called.setdefault("hit", ["ok"]))
    assert pdf_layout.extract_pdf_pages("whatever.pdf") == ["ok"]
    assert called["hit"] == ["ok"]
