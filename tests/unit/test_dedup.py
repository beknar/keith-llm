from keith_llm.data.dedup import exact_dedup, minhash_dedup, paragraph_dedup


def _rec(text, **extra):
    return {"text": text, **extra}


def _doc(mutate_at=(), extra_words=0):
    # Non-repeating word stream so the shingle set is large and mutations
    # move estimated Jaccard predictably.
    words = [f"lore{i}" for i in range(300)]
    for i in mutate_at:
        words[i] = f"changed{i}"
    words += [f"extra{i}" for i in range(extra_words)]
    return " ".join(words)


def test_exact_dedup_ignores_case_and_whitespace():
    records = [_rec("The Goblin  King"), _rec("the goblin king"), _rec("a different doc")]
    assert len(exact_dedup(records)) == 2


def test_paragraph_dedup_removes_boilerplate():
    ogl = (
        "OPEN GAME LICENSE Version 1.0a The following text is the property of "
        "Wizards of the Coast, Inc. and is Copyright 2000 Wizards of the Coast, Inc."
    )
    records = [_rec(f"Unique adventure content number {i}.\n\n{ogl}") for i in range(5)]
    records.append(_rec("A wholly unrelated document about sailing rules."))
    out = paragraph_dedup(records, max_docs=3)
    assert len(out) == 6
    assert all("OPEN GAME LICENSE" not in r["text"] for r in out)
    assert any("sailing rules" in r["text"] for r in out)


def test_paragraph_dedup_drops_emptied_docs():
    ogl = "x" * 100
    records = [_rec(ogl) for _ in range(5)]
    assert paragraph_dedup(records, max_docs=3) == []


def test_minhash_drops_near_duplicate_keeps_longer():
    near_a = _doc()
    near_b = _doc(mutate_at=(40,), extra_words=5)  # ~95% Jaccard, longer
    other = "completely different text about naval combat and boarding actions " * 20
    out = minhash_dedup([_rec(near_a, name="a"), _rec(near_b, name="b"), _rec(other, name="c")])
    names = {r["name"] for r in out}
    assert names == {"b", "c"}


def test_minhash_keeps_distinct_docs():
    docs = [
        "the wizard tower rises over the marsh " * 30,
        "goblin raiders strike the caravan at midnight " * 30,
        "the sunken temple hides a sleeping serpent god " * 30,
    ]
    assert len(minhash_dedup([_rec(d) for d in docs])) == 3


def test_minhash_deterministic():
    records = [_rec(_doc()), _rec(_doc(mutate_at=(5,), extra_words=3)), _rec("other content " * 50)]
    out1 = minhash_dedup([dict(r) for r in records])
    out2 = minhash_dedup([dict(r) for r in records])
    assert [r["text"] for r in out1] == [r["text"] for r in out2]
