import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from keith_llm.data import archive
from keith_llm.data.archive import extracted_documents, is_archive
from keith_llm.data.ingest import SUPPORTED_EXTS

PROSE = (
    b"The village of Emberfall sits at the edge of the mirewood, its palisade "
    b"scarred by last winter's raids and haunted by the barrow beneath the mill."
)


def _write_zip(path: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)


def _write_tar(path: Path, files: dict[str, bytes], mode: str = "w:gz") -> None:
    with tarfile.open(path, mode) as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


# --- is_archive ---


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("pack.zip", True),
        ("pack.tar", True),
        ("pack.tar.gz", True),
        ("pack.tgz", True),
        ("pack.tar.bz2", True),
        ("pack.tbz2", True),
        ("pack.tar.xz", True),
        ("pack.txz", True),
        ("Book.pdf.gz", True),
        ("notes.txt.bz2", True),
        ("data.xz", True),
        ("adventure.pdf", False),
        ("rules.txt", False),
        ("notes.docx", False),
        ("map.png", False),
    ],
)
def test_is_archive(name, expected):
    assert is_archive(Path(name)) is expected


# --- extraction: format coverage ---


def test_zip_extracts_supported_skips_others(tmp_path, make_pdf):
    pdf = make_pdf(["The goblin ambush begins at dusk."])
    z = tmp_path / "pack.zip"
    _write_zip(z, {"CoS.pdf": pdf, "rules.txt": PROSE, "notes.docx": b"junk", "cover.png": b"x"})
    with extracted_documents(z, SUPPORTED_EXTS) as members:
        names = sorted(n for n, _ in members)
        assert names == ["CoS.pdf", "rules.txt"]
        contents = {n: p.read_bytes() for n, p in members}
    assert contents["rules.txt"] == PROSE


def test_targz_extracts_members(tmp_path):
    t = tmp_path / "pack.tar.gz"
    _write_tar(t, {"a.txt": PROSE, "b.md": b"# Heading\n\nbody text here", "c.docx": b"no"})
    with extracted_documents(t, SUPPORTED_EXTS) as members:
        assert sorted(n for n, _ in members) == ["a.txt", "b.md"]


def test_tar_plain_and_xz(tmp_path):
    for suffix, mode in ((".tar", "w"), (".tar.xz", "w:xz"), (".tbz2", "w:bz2")):
        t = tmp_path / f"pack{suffix}"
        _write_tar(t, {"doc.txt": PROSE}, mode=mode)
        with extracted_documents(t, SUPPORTED_EXTS) as members:
            assert [n for n, _ in members] == ["doc.txt"]


def test_single_file_gz_txt(tmp_path):
    import gzip

    g = tmp_path / "Adventure.txt.gz"
    g.write_bytes(gzip.compress(PROSE))
    with extracted_documents(g, SUPPORTED_EXTS) as members:
        assert len(members) == 1
        name, mpath = members[0]
        assert name == "Adventure.txt"
        assert mpath.read_bytes() == PROSE


def test_single_file_gz_pdf(tmp_path, make_pdf):
    import gzip

    g = tmp_path / "Module.pdf.gz"
    g.write_bytes(gzip.compress(make_pdf(["Roll for initiative."])))
    with extracted_documents(g, SUPPORTED_EXTS) as members:
        assert [n for n, _ in members] == ["Module.pdf"]


def test_single_file_gz_unsupported_inner(tmp_path):
    import gzip

    g = tmp_path / "data.json.gz"
    g.write_bytes(gzip.compress(b'{"not": "a document"}'))
    with extracted_documents(g, SUPPORTED_EXTS) as members:
        assert members == []


def test_nested_archive_is_ignored(tmp_path):
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("deep.txt", PROSE)
    outer = tmp_path / "outer.zip"
    _write_zip(outer, {"inner.zip": inner.getvalue(), "top.txt": PROSE})
    with extracted_documents(outer, SUPPORTED_EXTS) as members:
        # inner.zip is skipped (its extension isn't a document type)
        assert [n for n, _ in members] == ["top.txt"]


# --- safety ---


def test_zip_slip_paths_are_neutralized(tmp_path):
    z = tmp_path / "evil.zip"
    _write_zip(z, {"../escape.txt": PROSE, "/abs/escape.txt": b"absolute payload here now"})
    with extracted_documents(z, SUPPORTED_EXTS) as members:
        # Labels preserve the malicious names, but destinations are safe temp
        # files this module named — nothing is written to the traversal target.
        assert {n for n, _ in members} == {"../escape.txt", "/abs/escape.txt"}
        for _, mpath in members:
            assert mpath.name.startswith("m") and mpath.suffix == ".txt"
            assert mpath.parent.name.startswith("keith_arch_")
    assert not (tmp_path.parent / "escape.txt").exists()
    assert not Path("/abs/escape.txt").exists()


def test_decompression_bomb_member_is_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "_MAX_MEMBER_BYTES", 16)
    z = tmp_path / "bomb.zip"
    _write_zip(z, {"big.txt": b"x" * 1000, "small.txt": b"tiny"})
    with extracted_documents(z, SUPPORTED_EXTS) as members:
        assert [n for n, _ in members] == ["small.txt"]  # oversized member dropped


def test_corrupt_archive_yields_nothing_or_raises(tmp_path):
    bad = tmp_path / "broken.zip"
    bad.write_bytes(b"this is not a zip file")
    with pytest.raises(Exception):  # noqa: B017 - build_corpus wraps this; here we just confirm it raises
        with extracted_documents(bad, SUPPORTED_EXTS) as members:
            list(members)
