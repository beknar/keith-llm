from keith_llm.data.clean import clean_pages, clean_text, is_quality


def test_clean_pages_strips_repeated_headers():
    bodies = [
        "The goblin waits.",
        "A wyvern circles.",
        "The door is trapped.",
        "Rain floods the crypt.",
        "An idol glimmers.",
        "The bridge collapses.",
    ]
    pages = [f"DUNGEON DELVER'S GUIDE\nPage {i}\n{body}" for i, body in enumerate(bodies, 1)]
    joined = clean_pages(pages)
    assert "DUNGEON DELVER'S GUIDE" not in joined
    assert "Page 3" not in joined  # digit-normalized page numbers count as repeats
    assert "The door is trapped." in joined


def test_clean_pages_leaves_short_docs_alone():
    pages = ["Header\nBody one.", "Header\nBody two."]
    joined = clean_pages(pages)
    assert joined.count("Header") == 2


def test_clean_text_dehyphenates_line_wraps():
    assert "fireball" in clean_text("the wizard casts fire-\nball at the horde")


def test_clean_text_fixes_mojibake_and_ligatures():
    cleaned = clean_text("the ﬁghter said â€œhaltâ€")
    assert "fighter" in cleaned  # NFKC folds the ﬁ ligature
    assert "“halt”" in cleaned or '"halt"' in cleaned  # ftfy repairs the mojibake


def test_clean_text_collapses_blank_runs():
    assert clean_text("a\n\n\n\n\nb") == "a\n\nb"


def test_is_quality_rejects_short_text():
    assert not is_quality("Too short.")


def test_is_quality_accepts_prose():
    prose = (
        "The party descends into the barrow, torchlight flickering over runes "
        "older than the kingdom itself. Something stirs in the dark below, and "
        "the air grows cold as the first skeleton claws free of its cairn."
    ) * 2
    assert is_quality(prose)


def test_is_quality_rejects_digit_soup():
    table = " ".join(str(n) for n in range(150))
    assert not is_quality(table)
