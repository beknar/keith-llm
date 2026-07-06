import json

from keith_llm.data.dedup_report import (
    apply_removals,
    build_report,
    find_overlaps,
    report_corpus,
)

# Distinct 5-word content so shingle sets are large and overlap is meaningful.
BASE = " ".join(f"lore{i}" for i in range(300))
EXTRA = " ".join(f"extra{i}" for i in range(2000))


def _rec(source, text, **kw):
    return {"source": source, "text": text, "system": "dnd5e", "doc_type": "adventure", **kw}


def test_containment_detected_despite_low_jaccard():
    # A (small) is almost entirely contained in B (much larger). Overlap
    # coefficient is ~1.0; Jaccard is small because B dwarfs A. The report must
    # flag this — MinHash/LSH keyed on Jaccard would not.
    records = [_rec("small.txt", BASE), _rec("big.pdf", BASE + " " + EXTRA)]
    pairs, sizes, _ = find_overlaps(records, threshold=0.75)
    assert len(pairs) == 1
    i, j, overlap, jaccard = pairs[0]
    assert overlap > 0.95
    assert jaccard < 0.75  # the whole point: containment, not similarity


def test_distinct_documents_have_no_pairs():
    records = [
        _rec("a.txt", " ".join(f"alpha{i}" for i in range(300))),
        _rec("b.txt", " ".join(f"beta{i}" for i in range(300))),
        _rec("c.txt", " ".join(f"gamma{i}" for i in range(300))),
    ]
    pairs, _, _ = find_overlaps(records, threshold=0.75)
    assert pairs == []


def test_keep_policy_drops_pdf_over_text():
    # Same content from two sources: keep the clean .txt, drop the .pdf.
    records = [_rec("CoS.pdf", BASE), _rec("CoS.txt", BASE)]
    report = build_report(records, threshold=0.75)
    assert report["n_clusters"] == 1
    cluster = report["clusters"][0]
    assert cluster["keep"]["source"] == "CoS.txt"
    assert [d["source"] for d in cluster["drop"]] == ["CoS.pdf"]
    assert report["drop_files"] == ["CoS.pdf"]


def test_cluster_of_three_keeps_one():
    records = [
        _rec("a.pdf", BASE),
        _rec("b.pdf", BASE + " tiny tail one"),
        _rec("c.txt", BASE + " tiny tail two"),
    ]
    report = build_report(records, threshold=0.75)
    assert report["n_clusters"] == 1
    cluster = report["clusters"][0]
    assert cluster["keep"]["source"] == "c.txt"  # non-PDF wins
    assert len(cluster["drop"]) == 2
    assert set(report["drop_files"]) == {"a.pdf", "b.pdf"}


def test_threshold_is_respected():
    # ~50% overlap: flagged at 0.4, not at 0.75.
    half = " ".join(f"lore{i}" for i in range(150))
    records = [
        _rec("x.txt", BASE),
        _rec("y.txt", half + " " + " ".join(f"z{i}" for i in range(150))),
    ]
    assert find_overlaps(records, threshold=0.75)[0] == []
    assert len(find_overlaps(records, threshold=0.40)[0]) == 1


def test_max_doc_frequency_cap_is_tunable():
    # Four identical docs: their shared shingles appear in all 4. With the cap
    # below 4 those shingles are skipped (boilerplate assumption) and the dupes
    # are missed; raising the cap catches them. Documents the exactness caveat.
    records = [_rec(f"copy{i}.txt", BASE) for i in range(4)]
    assert find_overlaps(records, threshold=0.75, max_doc_frequency=3)[0] == []
    assert len(find_overlaps(records, threshold=0.75, max_doc_frequency=10)[0]) == 6  # C(4,2)


def test_report_corpus_roundtrip(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    records = [_rec("keep.txt", BASE), _rec("dupe.pdf", BASE), _rec("unique.txt", EXTRA)]
    corpus.write_text("".join(json.dumps(r) + "\n" for r in records))
    report = report_corpus(corpus, threshold=0.75)
    assert report["n_documents"] == 3
    assert report["drop_files"] == ["dupe.pdf"]


def test_apply_quarantine_moves_drop_file(tmp_path):
    (tmp_path / "data/raw/dnd5e/adventure").mkdir(parents=True)
    keep = tmp_path / "data/raw/dnd5e/adventure/CoS.txt"
    drop = tmp_path / "data/raw/dnd5e/adventure/CoS.pdf"
    keep.write_text(BASE)
    drop.write_text(BASE)

    res = apply_removals(["data/raw/dnd5e/adventure/CoS.pdf"], root=tmp_path, hard=False)
    assert res["removed"] == ["data/raw/dnd5e/adventure/CoS.pdf"]
    assert not drop.exists()  # moved out of the ingest tree
    assert keep.exists()  # keeper untouched
    assert (tmp_path / "data/quarantine/data/raw/dnd5e/adventure/CoS.pdf").exists()


def test_apply_hard_deletes(tmp_path):
    f = tmp_path / "junk.pdf"
    f.write_text(BASE)
    res = apply_removals(["junk.pdf"], root=tmp_path, hard=True)
    assert res["removed"] == ["junk.pdf"]
    assert not f.exists()
    assert not (tmp_path / "data/quarantine").exists()


def test_apply_reports_missing_files(tmp_path):
    res = apply_removals(["gone.pdf"], root=tmp_path)
    assert res["removed"] == []
    assert res["missing"] == ["gone.pdf"]
