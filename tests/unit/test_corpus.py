import json
from pathlib import Path

import pytest

from keith_llm.data import corpus as corpus_mod
from keith_llm.data.corpus import build_corpus, filter_publishable, load_manifest

REPO_MANIFEST = Path(__file__).resolve().parents[2] / "data" / "sources.yaml"

PROSE = (
    "The village of Emberfall sits at the edge of the mirewood, its palisade "
    "scarred by last winter's raids. The elders speak of a barrow beneath the "
    "old mill where something ancient turns in its sleep, and they will pay "
    "good silver to anyone brave or foolish enough to see it stays sleeping."
)


def test_repo_manifest_is_valid():
    specs = load_manifest(REPO_MANIFEST)
    assert specs, "repo manifest should define sources"
    assert any(s.publishable for s in specs)
    assert any(not s.publishable for s in specs)


def test_manifest_rejects_unknown_system(tmp_path):
    m = tmp_path / "m.yaml"
    m.write_text(
        "sources:\n"
        "  - glob: 'x/**/*'\n    system: gurps\n    doc_type: rules\n"
        "    license: proprietary\n    publishable: false\n"
    )
    with pytest.raises(ValueError, match="gurps"):
        load_manifest(m)


def test_manifest_rejects_missing_keys(tmp_path):
    m = tmp_path / "m.yaml"
    m.write_text("sources:\n  - glob: 'x/**/*'\n    system: dnd5e\n")
    with pytest.raises(ValueError, match="missing"):
        load_manifest(m)


def test_filter_publishable():
    records = [{"publishable": True, "id": 1}, {"publishable": False, "id": 2}]
    assert [r["id"] for r in filter_publishable(records)] == [1]


def test_build_corpus_end_to_end(tmp_path):
    (tmp_path / "docs" / "dnd").mkdir(parents=True)
    (tmp_path / "docs" / "sw").mkdir(parents=True)
    (tmp_path / "docs" / "dnd" / "adv1.txt").write_text(PROSE)
    (tmp_path / "docs" / "dnd" / "adv1_copy.txt").write_text(PROSE)  # exact dup
    (tmp_path / "docs" / "dnd" / "short.txt").write_text("Too short.")  # low quality
    (tmp_path / "docs" / "sw" / "rules.md").write_text(
        PROSE.replace("Emberfall", "Redwater").replace("barrow", "sinkhole") + " Draw a card."
    )
    (tmp_path / "docs" / "dnd" / "notes.docx").write_text("ignored: unsupported type")
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        "sources:\n"
        "  - glob: 'docs/dnd/**/*'\n    system: dnd5e\n    doc_type: adventure\n"
        "    license: CC-BY-4.0\n    publishable: true\n"
        "  - glob: 'docs/sw/**/*'\n    system: savage_worlds\n    doc_type: rules\n"
        "    license: proprietary\n    publishable: false\n"
    )

    out = tmp_path / "corpus.jsonl"
    stats = build_corpus(manifest, out, root=tmp_path)

    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert stats["documents"] == len(records) == 2
    assert stats["dropped_exact_dup"] == 1
    assert stats["dropped_low_quality"] == 1
    assert stats["files_scanned"] == 4  # 4 supported docs attempted
    assert stats["skipped_unsupported"] == 1  # the .docx, now logged/counted

    by_system = {r["system"]: r for r in records}
    assert by_system["dnd5e"]["publishable"] is True
    assert by_system["dnd5e"]["license"] == "CC-BY-4.0"
    assert by_system["savage_worlds"]["publishable"] is False
    assert all(len(r["id"]) == 40 for r in records)
    assert by_system["dnd5e"]["source"].startswith("docs/dnd/")


def _one_txt_manifest(root: Path, text: str) -> Path:
    (root / "docs").mkdir(exist_ok=True)
    (root / "docs" / "book.txt").write_text(text)
    manifest = root / "m.yaml"
    manifest.write_text(
        "sources:\n"
        "  - glob: 'docs/**/*'\n    system: dnd5e\n    doc_type: rules\n"
        "    license: CC-BY-4.0\n    publishable: true\n"
    )
    return manifest


def test_reingest_uses_cache_and_skips_extraction(tmp_path, monkeypatch):
    manifest = _one_txt_manifest(tmp_path, PROSE)
    out = tmp_path / "corpus.jsonl"

    first = build_corpus(manifest, out, root=tmp_path)
    assert first["cache_hits"] == 0  # cold cache: extracted
    assert first["documents"] == 1

    # On the second run, extraction MUST NOT be called — the cache serves it.
    def boom(*a, **k):
        raise AssertionError("extract_pages must not run for a cached unchanged file")

    monkeypatch.setattr(corpus_mod, "extract_pages", boom)
    second = build_corpus(manifest, out, root=tmp_path)
    assert second["cache_hits"] == 1
    assert second["documents"] == 1
    # identical corpus produced from cache
    assert out.read_text() == (tmp_path / "corpus.jsonl").read_text()


def test_changed_file_is_reextracted(tmp_path):
    manifest = _one_txt_manifest(tmp_path, PROSE)
    out = tmp_path / "corpus.jsonl"
    build_corpus(manifest, out, root=tmp_path)

    # Modify the file's content -> new hash -> cache miss -> re-extracted.
    (tmp_path / "docs" / "book.txt").write_text(
        PROSE.replace("Emberfall", "Duskholt").replace("barrow", "cavern")
    )
    stats = build_corpus(manifest, out, root=tmp_path)
    assert stats["cache_hits"] == 0
    text = json.loads(out.read_text())["text"]
    assert "Duskholt" in text


def test_no_cache_flag_bypasses_cache(tmp_path):
    manifest = _one_txt_manifest(tmp_path, PROSE)
    out = tmp_path / "corpus.jsonl"
    build_corpus(manifest, out, root=tmp_path, use_cache=True)
    stats = build_corpus(manifest, out, root=tmp_path, use_cache=False)
    assert stats["cache_hits"] == 0  # cache ignored even though an entry exists


def test_corrupt_cache_degrades_gracefully(tmp_path):
    manifest = _one_txt_manifest(tmp_path, PROSE)
    out = tmp_path / "corpus.jsonl"
    cache_file = tmp_path / "data" / "cache" / "extraction.sqlite"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(b"not a sqlite database at all")
    stats = build_corpus(manifest, out, root=tmp_path)
    assert stats["documents"] == 1  # still ingests, just without the cache


def test_build_corpus_expands_archives(tmp_path):
    import zipfile

    (tmp_path / "docs").mkdir()
    unique = PROSE.replace("Emberfall", "Duskwater").replace("mill", "quarry")
    zpath = tmp_path / "docs" / "pack.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("adventures/A.txt", PROSE)
        zf.writestr("adventures/B.txt", unique)
        zf.writestr("cover.png", b"not a document")
    (tmp_path / "docs" / "loose.rtf").write_text("unsupported loose file")

    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        "sources:\n"
        "  - glob: 'docs/**/*'\n    system: dnd5e\n    doc_type: adventure\n"
        "    license: CC-BY-4.0\n    publishable: true\n"
    )
    out = tmp_path / "corpus.jsonl"
    stats = build_corpus(manifest, out, root=tmp_path)
    records = [json.loads(line) for line in out.read_text().splitlines()]

    assert stats["archives_expanded"] == 1
    assert stats["skipped_unsupported"] == 1  # loose.rtf
    assert stats["files_scanned"] == 2  # two .txt members (cover.png not a doc)
    assert len(records) == 2
    sources = sorted(r["source"] for r in records)
    assert sources == ["docs/pack.zip!adventures/A.txt", "docs/pack.zip!adventures/B.txt"]
    assert all(r["system"] == "dnd5e" and r["publishable"] for r in records)
