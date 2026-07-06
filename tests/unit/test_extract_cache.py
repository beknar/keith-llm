from keith_llm.data import extract_cache
from keith_llm.data.extract_cache import (
    ExtractionCache,
    current_version,
    hash_file,
)


def test_hash_file_content_addressed(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    c = tmp_path / "c.txt"
    a.write_bytes(b"the same bytes")
    b.write_bytes(b"the same bytes")
    c.write_bytes(b"different bytes")
    assert hash_file(a) == hash_file(b)  # identical content -> identical key
    assert hash_file(a) != hash_file(c)
    assert len(hash_file(a)) == 64  # sha256 hex


def test_cache_roundtrip(tmp_path):
    cache = ExtractionCache(tmp_path / "cache" / "x.sqlite", current_version(True))
    assert cache.get("deadbeef") is None
    cache.put("deadbeef", "extracted text")
    assert cache.get("deadbeef") == "extracted text"
    cache.close()


def test_cache_persists_across_reopen(tmp_path):
    path = tmp_path / "c.sqlite"
    v = current_version(True)
    c1 = ExtractionCache(path, v)
    c1.put("h1", "hello")
    c1.close()
    c2 = ExtractionCache(path, v)
    assert c2.get("h1") == "hello"
    c2.close()


def test_version_isolates_entries(tmp_path):
    path = tmp_path / "c.sqlite"
    with_ocr = ExtractionCache(path, current_version(True))
    with_ocr.put("h", "ocr text")
    with_ocr.close()
    # A different version (e.g. OCR now applied differently) must not hit.
    without_ocr = ExtractionCache(path, current_version(False))
    assert without_ocr.get("h") is None
    without_ocr.put("h", "no-ocr text")  # coexists, doesn't clobber
    without_ocr.close()
    again = ExtractionCache(path, current_version(True))
    assert again.get("h") == "ocr text"  # original entry intact
    again.close()


def test_current_version_distinct():
    assert current_version(True) != current_version(False)


def test_current_version_includes_tool_versions():
    v = current_version(False)
    assert "pypdf=" in v and "pdfplumber=" in v


def test_tool_upgrade_invalidates_cache(tmp_path, monkeypatch):
    # A file cached under one pdfplumber version must miss after an "upgrade",
    # so a changed extractor can't serve stale text.
    path = tmp_path / "c.sqlite"
    monkeypatch.setattr(
        extract_cache, "_dep_version", lambda name: "1.0" if name == "pdfplumber" else "x"
    )
    c1 = ExtractionCache(path, current_version(False))
    c1.put("h", "old-extractor text")
    c1.close()

    monkeypatch.setattr(
        extract_cache, "_dep_version", lambda name: "2.0" if name == "pdfplumber" else "x"
    )
    c2 = ExtractionCache(path, current_version(False))
    assert c2.get("h") is None  # extractor upgraded -> cache miss -> re-extract
    c2.close()
