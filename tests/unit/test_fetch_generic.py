from pathlib import Path

from keith_llm.constants import DOC_TYPES
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


def test_write_shards_rerun_replaces_not_merges(tmp_path):
    # first run: many small shards
    write_shards([LONG] * 6, tmp_path, "src", shard_bytes=10)
    assert len(list(tmp_path.glob("src-*.txt"))) == 6
    # re-run with fewer docs must not leave orphaned higher-numbered shards
    write_shards([LONG], tmp_path, "src", shard_bytes=10)
    assert {f.name for f in tmp_path.glob("src-*.txt")} == {"src-00001.txt"}
    # a differently-named source in the same dir is untouched
    write_shards([LONG], tmp_path, "other", shard_bytes=10)
    write_shards([LONG], tmp_path, "src", shard_bytes=10)
    assert (tmp_path / "other-00001.txt").exists()


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


def test_cmd_fetch_generic_hard_exits_after_success(tmp_path, monkeypatch):
    # the command hard-exits to dodge the datasets/fsspec shutdown crash; verify
    # it reaches that hard-exit on success (patched so it doesn't kill pytest)
    import argparse

    import keith_llm.cli as cli
    import keith_llm.data.fetch_generic as fg

    calls = {"exit": 0}
    monkeypatch.setattr(cli, "_hard_exit_after_fetch", lambda: calls.__setitem__("exit", 1))
    monkeypatch.setattr(fg, "fetch_generic", lambda *a, **k: {"wikitext": {"docs": 3}})

    args = argparse.Namespace(
        list=False,
        repo_id=None,
        sources=["wikitext"],
        out=str(tmp_path),
        max_mb=100.0,
        max_docs=None,
    )
    assert cli._cmd_fetch_generic(args) == 0
    assert calls["exit"] == 1  # hard-exit was invoked after writing/printing


def test_recipes_are_valid_and_tagged_generic():
    assert {"wikitext", "wikipedia", "gutenberg"} <= set(RECIPES)
    for r in RECIPES.values():
        assert r.doc_type in DOC_TYPES  # must be a real control-token doc_type
        assert r.repo_id and r.text_column


def test_manifest_registers_generic_sources(tmp_path):
    # the shipped manifest must register the subdirs the fetcher writes. Scope to
    # the openly-licensed fetch outputs under data/seed/generic/ — other
    # system: generic entries (e.g. proprietary data/raw/other) are unrelated.
    specs = load_manifest("data/sources.yaml")
    seed_generic = [s for s in specs if "data/seed/generic/" in s.glob]
    for s in seed_generic:
        assert s.system == "generic" and s.doc_type in DOC_TYPES
        assert s.publishable is True  # openly licensed

    # and each recipe's actual output must be matched by its manifest glob (not
    # just a substring check) — a real file written under the subdir is globbed
    for recipe in RECIPES.values():
        spec = next((s for s in seed_generic if f"generic/{recipe.subdir}/" in s.glob), None)
        assert spec is not None, f"no manifest entry for {recipe.subdir}"
        fetch_one(recipe, recipe.subdir, tmp_path, reader=_reader([LONG]))
        written = tmp_path / recipe.subdir / f"{recipe.subdir}-00001.txt"
        assert written.exists()
        # the manifest glob is repo-relative (data/seed/generic/<sub>/**/*);
        # re-root it at tmp_path and confirm it actually matches the written file
        rel = spec.glob.replace("data/seed/generic/", "")
        assert written in set(tmp_path.glob(rel))
