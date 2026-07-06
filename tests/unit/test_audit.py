import json

from keith_llm.data.audit import audit_corpus, score_text

GOOD_PROSE = (
    "The village of Emberfall sits at the edge of the mirewood, its palisade "
    "scarred by last winter's raids. The elders speak of a barrow beneath the "
    "old mill where something ancient turns in its sleep, and they will pay "
    "good silver to anyone brave enough to see that it stays sleeping. "
) * 4

# Column interleaving smashes word boundaries together, producing internal
# case transitions ("damageThe").
INTERLEAVED = (
    "damageThe wizardCast fireballAt theGoblin hordeRolling twoSaves "
    "againstThe spellThey mostlyFailed andBurned toAsh quicklyNow "
) * 6

# A broken text layer yields vowelless gibberish.
GIBBERISH = "brtsk wxqz fghj mntp zzzk vvbb qwrt plld " * 20

# pypdf's multi-column failure: real words, intact, but one per line and in
# scrambled order. Token metrics look fine; words-per-line is the tell.
WORD_SALAD = "\n".join(
    ["The", "village", "of", "Emberfall", "sits", "at", "the", "edge", "of", "mirewood"] * 40
)

# Dense stat-block content: low alpha ratio, but legitimately extracted.
STATBLOCK = (
    "Mire Goblin\nSmall humanoid, neutral evil\nArmor Class 15\n"
    "Hit Points 7 (2d6)\nSpeed 30 ft.\nSTR 8 DEX 14 CON 10 INT 10 WIS 8 CHA 8\n"
    "Challenge 1/4\nThe goblin makes one scimitar attack against a target it can "
    "see, dealing five slashing damage on a hit, and then retreats into the reeds. "
) * 4


def test_good_prose_is_ok():
    assert score_text(GOOD_PROSE)["verdict"] == "OK"


def test_interleaved_text_is_bad():
    m = score_text(INTERLEAVED)
    assert m["internal_caps_rate"] > 0.15
    assert m["verdict"] == "BAD"


def test_gibberish_is_bad():
    m = score_text(GIBBERISH)
    assert m["wordlike_frac"] < 0.60
    assert m["verdict"] == "BAD"


def test_word_salad_flagged_despite_intact_words():
    # The column-failure case that token metrics alone miss: every word is a
    # real word, but one-per-line reading order is broken.
    m = score_text(WORD_SALAD)
    assert m["wordlike_frac"] > 0.80  # words themselves are fine
    assert m["words_per_line"] < 2.0
    assert m["verdict"] == "BAD"


def test_letter_spaced_extraction_is_bad():
    # Per-glyph positioning that pdfplumber splits into single characters: many
    # tokens, but none form a length>=2 word. Must not default to a healthy OK.
    m = score_text("T h e d a m a g e d d o o r s w i n g s o p e n " * 30)
    assert m["wordlike_frac"] == 0.0
    assert m["verdict"] == "BAD"


def test_statblock_not_flagged_bad():
    # Dense tables/numbers must not be mistaken for a broken extraction.
    assert score_text(STATBLOCK)["verdict"] in ("OK", "WARN")


def test_metrics_present_and_typed():
    m = score_text(GOOD_PROSE)
    for key in (
        "n_chars",
        "n_tokens",
        "alpha_ratio",
        "wordlike_frac",
        "internal_caps_rate",
        "long_token_frac",
        "garbage_line_frac",
    ):
        assert key in m
    assert 0.0 <= m["alpha_ratio"] <= 1.0


def test_empty_text_does_not_crash():
    m = score_text("")
    assert m["n_tokens"] == 0


def test_audit_corpus_orders_worst_first(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    records = [
        {"source": "good.txt", "system": "dnd5e", "doc_type": "adventure", "text": GOOD_PROSE},
        {"source": "bad.pdf", "system": "dnd5e", "doc_type": "rules", "text": INTERLEAVED},
        {"source": "junk.pdf", "system": "d6", "doc_type": "rules", "text": GIBBERISH},
    ]
    corpus.write_text("".join(json.dumps(r) + "\n" for r in records))

    report = audit_corpus(corpus)
    assert report["n_documents"] == 3
    assert report["verdicts"].get("BAD") == 2
    assert report["verdicts"].get("OK") == 1
    # worst first: the two BAD docs precede the OK one
    verdicts = [d["verdict"] for d in report["documents"]]
    assert verdicts == ["BAD", "BAD", "OK"]
    assert report["documents"][0]["source"] in {"bad.pdf", "junk.pdf"}
    assert report["documents"][-1]["source"] == "good.txt"
