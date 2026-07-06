"""Extract ingestible documents from archive / compressed files.

Supports ``.zip``, tar in any form (``.tar``, ``.tar.gz``/``.tgz``,
``.tar.bz2``/``.tbz2``, ``.tar.xz``/``.txz``), and single-file compressors
(``.gz``/``.bz2``/``.xz``) that wrap one document, e.g. ``Adventure.pdf.gz``.
Only members whose own extension is a supported document type are pulled out;
nested archives are therefore ignored (their ``.zip``/``.tar`` extension isn't
a document type), which also bounds recursion.

Security: member-supplied paths are NEVER used as write destinations — each
member is read and copied to a temp filename this module chooses — so
path-traversal ("zip-slip", malicious ``..``/absolute tar members) cannot
write outside the temp directory. Per-member output is capped to guard against
decompression bombs. Non-regular members (dirs, symlinks, devices) are skipped.
"""

from __future__ import annotations

import bz2
import gzip
import logging
import lzma
import shutil
import tarfile
import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

_TAR_SUFFIXES = {".tar", ".tgz", ".tbz2", ".tbz", ".txz"}
_SINGLE_OPENERS = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}
_MAX_MEMBER_BYTES = 512 * 1024 * 1024  # 512 MB per member (decompression-bomb guard)


def is_archive(path: Path) -> bool:
    suffix = path.suffix.lower()
    suffixes = [s.lower() for s in path.suffixes]
    if suffix == ".zip" or suffix in _TAR_SUFFIXES:
        return True
    if len(suffixes) >= 2 and suffixes[-2] == ".tar" and suffix in _SINGLE_OPENERS:
        return True  # .tar.gz / .tar.bz2 / .tar.xz
    return suffix in _SINGLE_OPENERS  # single-file compressor


def _doc_ext(name: str) -> str:
    return Path(name).suffix.lower()


def _copy_capped(src, dest: Path) -> bool:
    """Copy a binary stream to ``dest``, aborting if it exceeds the size cap.
    Returns True on success, False (and removes the partial file) if too large."""
    written = 0
    with dest.open("wb") as out:
        while chunk := src.read(1 << 20):
            written += len(chunk)
            if written > _MAX_MEMBER_BYTES:
                out.close()
                dest.unlink(missing_ok=True)
                return False
            out.write(chunk)
    return True


def _is_tar(path: Path) -> bool:
    suffix = path.suffix.lower()
    suffixes = [s.lower() for s in path.suffixes]
    if suffix in _TAR_SUFFIXES:
        return True
    return len(suffixes) >= 2 and suffixes[-2] == ".tar" and suffix in _SINGLE_OPENERS


@contextmanager
def extracted_documents(path: Path, supported_exts: set[str]) -> Iterator[list[tuple[str, Path]]]:
    """Extract supported documents from an archive into a temporary directory.

    Yields a list of ``(member_name, temp_path)``; ``member_name`` is the
    archive-internal path (for labelling only). The temp directory and its
    contents are removed when the context exits, so callers must finish reading
    the temp files inside the ``with`` block.
    """
    tmp = Path(tempfile.mkdtemp(prefix="keith_arch_"))
    out: list[tuple[str, Path]] = []
    try:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    ext = _doc_ext(info.filename)
                    if ext not in supported_exts:
                        continue
                    dest = tmp / f"m{len(out)}{ext}"
                    with zf.open(info) as src:
                        ok = _copy_capped(src, dest)
                    if ok:
                        out.append((info.filename, dest))
                    else:
                        logger.warning(
                            "archive member too large, skipping: %s!%s", path.name, info.filename
                        )
        elif _is_tar(path):
            with tarfile.open(path) as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    ext = _doc_ext(member.name)
                    if ext not in supported_exts:
                        continue
                    src = tf.extractfile(member)
                    if src is None:
                        continue
                    dest = tmp / f"m{len(out)}{ext}"
                    with src:
                        ok = _copy_capped(src, dest)
                    if ok:
                        out.append((member.name, dest))
                    else:
                        logger.warning(
                            "archive member too large, skipping: %s!%s", path.name, member.name
                        )
        else:  # single-file compressor
            opener = _SINGLE_OPENERS[path.suffix.lower()]
            inner = path.name[: -len(path.suffix)]  # strip the compression suffix
            ext = _doc_ext(inner)
            if ext in supported_exts:
                dest = tmp / f"m0{ext}"
                with opener(path, "rb") as src:
                    ok = _copy_capped(src, dest)
                if ok:
                    out.append((inner, dest))
                else:
                    logger.warning("compressed file too large, skipping: %s", path.name)
        yield out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
