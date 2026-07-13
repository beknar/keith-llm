from pathlib import Path

from keith_llm.constants import DOC_TYPES, SYSTEMS
from keith_llm.data.corpus import load_manifest
from keith_llm.data.fetch_generic import (
    RECIPES,
    Recipe,
    fetch_generic,
    fetch_one,
    write_shards,
)


# a canned reader that ignores its args and yields fixed documents (no network)
def _reader(docs):
    def read(repo_id, config, split, text_column):
        return iter(docs)

    return read


LONG = "word " * 40  # ~200 chars, comfortably over min_chars


# --- write_shards ---


def test_write_shards_writes_and_counts(tmp_path):
    stats = write_shards([LONG, LONG, LONG], tmp_path, "src", shard_bytes=10_000_000)
    assert stats["docs"] == 3
    assert stats["files"] == 1  # all fit in one shard
    files = sorted(tmp_path.glob("*.txt"))
    assert [f.name for f in files] == ["src-00001.txt"]
    assert files[0].read_text().count(LONG.strip()) == 3


def test_write_shards_rolls_over_at_shard_bytes(tmp_path):
    # tiny shard budget -> one file per doc
    stats = write_shards([LONG, LONG, LONG], tmp_path, "src", shard_bytes=10)
    assert stats["files"] == 3
    assert {f.name for f in tmp_path.glob("*.txt")} == {
        "src-00001.txt",
        "src-00002.txt",
        "src-00003.txt",
    }


def test_write_shards_respects_max_docs(tmp_path):
    stats = write_shards([LONG] * 100, tmp_path, "src", shard_bytes=10, max_docs=5)
    assert stats["docs"] == 5


def test_write_shards_respects_max_bytes(tmp_path):
    stats = write_shards([LONG] * 100, tmp_path, "src", shard_bytes=10, max_bytes=len(LONG.strip()))
    # stops as soon as total bytes reach the cap (after the first doc)
    assert stats["docs"] == 1


def test_write_shards_skips_short_docs(tmp_path):
    stats = write_shards(["hi", "", LONG, "  ", LONG], tmp_path, "src")
    assert stats["docs"] == 2  # only the two long ones


# --- fetch_one / fetch_generic with an injected reader ---


def test_fetch_one_writes_under_subdir(tmp_path):
    recipe = Recipe("acme/ds", "en", "train", "text", "public-domain", "setting", "mydata")
    stats = fetch_one(recipe, "mydata", tmp_path, reader=_reader([LONG, LONG]))
    assert stats["docs"] == 2
    assert stats["license"] == "public-domain"
    assert stats["repo_id"] == "acme/ds"
    assert Path(stats["path"]) == tmp_path / "mydata"
    assert (tmp_path / "mydata" / "mydata-00001.txt").exists()


def test_fetch_generic_multiple_named_sources(tmp_path):
    out = tmp_path / "generic"
    stats = fetch_generic(
        ["wikitext", "gutenberg"], out, max_mb=None, reader=_reader([LONG, LONG, LONG])
    )
    assert set(stats) == {"wikitext", "gutenberg"}
    assert stats["wikitext"]["docs"] == 3
    # each source lands in its own recipe subdir
    assert (out / "wikitext" / "wikitext-00001.txt").exists()
    assert (out / "gutenberg" / "gutenberg-00001.txt").exists()


def test_fetch_generic_unknown_source_errors(tmp_path):
    import pytest

    with pytest.raises(SystemExit, match="unknown source"):
        fetch_generic(["nope"], tmp_path, reader=_reader([LONG]))


def test_fetch_generic_max_mb_caps_bytes(tmp_path):
    # a huge stream but a tiny byte cap -> only the first doc is written
    stats = fetch_generic(
        ["wikitext"], tmp_path, max_mb=len(LONG.strip()) / 1_000_000, reader=_reader([LONG] * 1000)
    )
    assert stats["wikitext"]["docs"] == 1


# --- registry + manifest integration ---


def test_recipes_are_valid_and_tagged_generic():
    assert {"wikitext", "wikipedia", "gutenberg"} <= set(RECIPES)
    for r in RECIPES.values():
        assert r.doc_type in DOC_TYPES  # must be a real control-token doc_type
        assert r.repo_id and r.text_column


def test_manifest_registers_generic_sources():
    # the shipped manifest must include the generic subdirs the fetcher writes,
    # and they must validate (system/doc_type in the frozen vocab)
    specs = load_manifest("data/sources.yaml")
    generic = [s for s in specs if s.system == "generic"]
    subdirs = {r.subdir for r in RECIPES.values()}
    for sub in subdirs:
        assert any(f"generic/{sub}/" in s.glob for s in generic), f"no manifest entry for {sub}"
    for s in generic:
        assert s.system in SYSTEMS and s.doc_type in DOC_TYPES
        assert s.publishable is True  # openly licensed
