import io
import tarfile
import zipfile

from keith_llm.data import convert
from keith_llm.data.convert import (
    convert_file,
    convert_tree,
    fix_spacing,
    html_to_text,
    is_readable,
    reflow,
)

PROSE = (
    "The village of Emberfall sits at the edge of the mirewood, its palisade "
    "scarred by last winter's raids. The elders speak of a barrow beneath the old "
    "mill where something ancient turns in its sleep, and they will pay good silver "
    "to anyone brave enough to see that it stays sleeping forever more."
)


# --- HTML ---


def test_html_to_text():
    html = (
        "<html><head><style>x{}</style></head><body><h1>Title</h1><p>Hello there world.</p>"
        "<script>evil()</script></body></html>"
    )
    text = html_to_text(html)
    assert "Title" in text
    assert "Hello there world." in text
    assert "evil" not in text  # script stripped
    assert "x{}" not in text  # style stripped


# --- reflow ---


def test_reflow_joins_wrapped_lines():
    wrapped = "The goblin waits\nin the dark cave\nfor the party.\n\nA new paragraph\nhere."
    out = reflow(wrapped)
    assert out == "The goblin waits in the dark cave for the party.\n\nA new paragraph here."


# --- fix_spacing (needs wordninja; skip cleanly if absent) ---


def test_fix_spacing_splits_clumped_words():
    try:
        import wordninja  # noqa: F401
    except ImportError:
        return
    # a long run-together token gets split; short/normal words untouched
    out = fix_spacing("the thegoblinattacksthehero waits")
    assert "the goblin attacks the hero" in out
    assert out.startswith("the ")  # leading normal word preserved


def test_fix_spacing_leaves_normal_prose():
    try:
        import wordninja  # noqa: F401
    except ImportError:
        return
    assert fix_spacing(PROSE) == PROSE  # no >=19-char alpha runs to split


# --- readability gate ---


def test_is_readable_accepts_prose():
    assert is_readable(PROSE * 2)


def test_is_readable_rejects_short():
    assert not is_readable("too short")


def test_is_readable_rejects_gibberish():
    assert not is_readable("brtsk wxqz fghj mntp zzzk vvbb qwrt plld " * 30)


# --- convert_file dispatch ---


def test_convert_text_file(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text(PROSE * 2)
    assert PROSE.split(".")[0] in convert_file(f)


def test_convert_pdf(tmp_path, make_pdf):
    f = tmp_path / "doc.pdf"
    f.write_bytes(make_pdf([PROSE, PROSE.replace("Emberfall", "Duskwater")]))
    text = convert_file(f, enable_ocr=False)
    assert text and "mirewood" in text


def test_convert_html_file(tmp_path):
    f = tmp_path / "page.html"
    f.write_text(f"<html><body><p>{PROSE * 2}</p></body></html>")
    text = convert_file(f)
    assert text and "Emberfall" in text


def test_convert_rejects_unreadable(tmp_path):
    f = tmp_path / "junk.txt"
    f.write_text("zx qw zz " * 60)  # gibberish -> discarded
    assert convert_file(f) is None


def test_convert_unsupported_type(tmp_path):
    f = tmp_path / "clip.m4v"
    f.write_bytes(b"\x00\x00fake video")
    assert convert_file(f) is None


# --- convert_tree: recursion, archives, output structure ---


def _write_zip(path, files):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)


def _write_targz(path, files):
    with tarfile.open(path, "w:gz") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def test_convert_tree_mirrors_structure(tmp_path):
    src = tmp_path / "src"
    (src / "adv1").mkdir(parents=True)
    (src / "adv1" / "story.txt").write_text(PROSE * 2)
    (src / "adv1" / "notes.md").write_text(PROSE.replace("Emberfall", "Redwater") * 2)
    (src / "junk.m4v").write_bytes(b"video")  # skipped
    (src / "empty.txt").write_text("hi")  # rejected (too short)

    out = tmp_path / "out"
    stats = convert_tree(src, out, enable_ocr=False)
    assert stats["converted"] == 2
    assert stats["rejected"] == 1
    assert stats["skipped"] == 1
    # output name appends .txt to the full source name (collision-safe)
    assert (out / "adv1" / "story.txt.txt").read_text().startswith("The village of Emberfall")
    assert (out / "adv1" / "notes.md.txt").exists()
    assert not (out / "junk.txt").exists()


def test_convert_tree_no_same_stem_collision(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "doc.txt").write_text(PROSE * 2)
    (src / "doc.md").write_text(PROSE.replace("Emberfall", "Redwater") * 2)
    out = tmp_path / "out"
    convert_tree(src, out, enable_ocr=False)
    # both survive — no clobber (would collide if we replaced the suffix)
    assert (out / "doc.txt.txt").exists()
    assert (out / "doc.md.txt").exists()


def test_convert_tree_skips_symlinks(tmp_path):
    import os

    src = tmp_path / "src"
    src.mkdir()
    (src / "real.txt").write_text(PROSE * 2)
    # a directory symlink back at src would loop forever without the guard
    os.symlink(src, src / "loop", target_is_directory=True)
    out = tmp_path / "out"
    stats = convert_tree(src, out, enable_ocr=False)  # must return, not hang
    assert stats["converted"] == 1
    assert (out / "real.txt.txt").exists()


def test_convert_tree_skips_mac_junk(tmp_path):
    # Mac-made zips are full of __MACOSX/._* AppleDouble files with real
    # extensions but no data — skip them, don't try (and fail) to OCR them.
    src = tmp_path / "src"
    (src / "__MACOSX").mkdir(parents=True)
    (src / "__MACOSX" / "._art.jpg").write_bytes(b"not an image")
    (src / "._MD-1.tif").write_bytes(b"junk")
    (src / ".DS_Store").write_bytes(b"junk")
    (src / "real.txt").write_text(PROSE * 2)
    out = tmp_path / "out"
    stats = convert_tree(src, out, enable_ocr=False)
    assert stats["converted"] == 1
    assert stats["failed"] == 0  # junk skipped, not attempted -> no failures
    assert (out / "real.txt.txt").exists()


def test_convert_tree_rejects_non_directory_src(tmp_path):
    import pytest

    f = tmp_path / "notadir.txt"
    f.write_text("x")
    with pytest.raises(NotADirectoryError):
        convert_tree(f, tmp_path / "out")


# --- resume: skip files already converted by an earlier run ---


def test_convert_tree_resumes_from_existing_output(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text(PROSE * 2)
    (src / "b.txt").write_text(PROSE.replace("Emberfall", "Redwater") * 2)
    out = tmp_path / "out"

    first = convert_tree(src, out, enable_ocr=False)
    assert first["converted"] == 2
    assert first["cached"] == 0

    # a re-run must not re-read anything: fail hard if convert_file is called
    def boom(*a, **k):
        raise AssertionError("convert_file should not run for a cached file")

    monkeypatch.setattr(convert, "convert_file", boom)
    second = convert_tree(src, out, enable_ocr=False)
    assert second["converted"] == 0
    assert second["cached"] == 2


def test_convert_tree_reconverts_when_source_is_newer(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    f = src / "a.txt"
    f.write_text(PROSE * 2)
    out = tmp_path / "out"
    convert_tree(src, out, enable_ocr=False)
    dest = out / "a.txt.txt"

    # edit the source and stamp it newer than the existing output
    f.write_text(PROSE.replace("Emberfall", "Newhaven") * 2)
    import os

    future = dest.stat().st_mtime + 100
    os.utime(f, (future, future))

    stats = convert_tree(src, out, enable_ocr=False)
    assert stats["converted"] == 1
    assert stats["cached"] == 0
    assert "Newhaven" in dest.read_text()


def test_convert_tree_force_reconverts_everything(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text(PROSE * 2)
    out = tmp_path / "out"
    convert_tree(src, out, enable_ocr=False)

    stats = convert_tree(src, out, enable_ocr=False, force=True)
    assert stats["converted"] == 1
    assert stats["cached"] == 0


def test_convert_tree_reconverts_when_output_empty(tmp_path):
    # a partial/empty .txt left by a killed write must not count as done
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text(PROSE * 2)
    out = tmp_path / "out"
    convert_tree(src, out, enable_ocr=False)
    dest = out / "a.txt.txt"
    dest.write_text("")  # simulate a truncated write

    stats = convert_tree(src, out, enable_ocr=False)
    assert stats["converted"] == 1
    assert stats["cached"] == 0


def test_convert_tree_expands_archives(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _write_zip(src / "pack.zip", {"inner/tale.txt": (PROSE * 2).encode(), "art.bin": b"x"})
    _write_targz(src / "bundle.tar.gz", {"lore.md": (PROSE.replace("mill", "keep") * 2).encode()})

    out = tmp_path / "out"
    stats = convert_tree(src, out, enable_ocr=False)
    assert stats["archives"] == 2
    assert stats["converted"] == 2
    # outputs nested under the archive filename
    assert (out / "pack.zip" / "inner" / "tale.txt.txt").exists()
    assert (out / "bundle.tar.gz" / "lore.md.txt").exists()


def test_extract_archive_rejects_bomb(tmp_path, monkeypatch):
    monkeypatch.setattr(convert, "_MAX_ARCHIVE_BYTES", 100)
    z = tmp_path / "bomb.zip"
    _write_zip(z, {"big.txt": b"x" * 5000})
    import pytest

    with pytest.raises(ValueError, match="inflates"):
        convert._extract_archive(z, tmp_path / "d")


def test_convert_tree_survives_bad_file(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "good.txt").write_text(PROSE * 2)
    (src / "bad.pdf").write_bytes(b"not a real pdf")  # extraction raises

    out = tmp_path / "out"
    stats = convert_tree(src, out, enable_ocr=False)
    # the good file still converts; the bad one is counted, not fatal
    assert stats["converted"] == 1
    assert stats["failed"] + stats["rejected"] == 1


def test_extract_archive_is_traversal_safe(tmp_path):
    z = tmp_path / "evil.zip"
    _write_zip(z, {"../escape.txt": PROSE.encode()})
    dest = tmp_path / "dest"
    dest.mkdir()
    convert._extract_archive(z, dest)
    assert not (tmp_path / "escape.txt").exists()  # did not escape dest
