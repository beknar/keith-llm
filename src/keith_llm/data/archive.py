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
write outside the temp directory. Non-regular members (dirs, symlinks,
devices) are skipped. Extraction is bounded three ways against decompression
bombs: a per-member size cap, an aggregate byte budget, and a member count
cap; hitting a limit stops extraction (logged) rather than filling the disk.
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
_MAX_MEMBER_BYTES = 512 * 1024 * 1024  # 512 MB per member
_MAX_TOTAL_BYTES = 4 * 1024 * 1024 * 1024  # 4 GB extracted per archive
_MAX_MEMBERS = 10_000  # ingestible members per archive


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


def _copy_capped(src, dest: Path, cap: int) -> int | None:
    """Copy a binary stream to ``dest``, aborting if it exceeds ``cap`` bytes.
    Returns bytes written, or None (removing the partial file) if the cap is
    exceeded. Checked incrementally, so an oversized member never fully lands."""
    written = 0
    with dest.open("wb") as out:
        while chunk := src.read(1 << 20):
            written += len(chunk)
            if written > cap:
                out.close()
                dest.unlink(missing_ok=True)
                return None
            out.write(chunk)
    return written


def _is_tar(path: Path) -> bool:
    suffix = path.suffix.lower()
    suffixes = [s.lower() for s in path.suffixes]
    if suffix in _TAR_SUFFIXES:
        return True
    return len(suffixes) >= 2 and suffixes[-2] == ".tar" and suffix in _SINGLE_OPENERS


def _accept(
    name: str,
    src,
    tmp: Path,
    supported_exts: set[str],
    out: list[tuple[str, Path]],
    state: dict[str, int],
    archive_name: str,
) -> bool:
    """Extract one member's stream if it is a supported document and within the
    extraction budget. Mutates ``out``/``state``. Returns False to tell the
    caller to STOP iterating (member-count or aggregate-byte limit reached)."""
    ext = _doc_ext(name)
    if ext not in supported_exts:
        return True
    if len(out) >= _MAX_MEMBERS:
        logger.warning(
            "archive %s exceeds %d ingestible members; stopping", archive_name, _MAX_MEMBERS
        )
        return False
    remaining = _MAX_TOTAL_BYTES - state["total"]
    dest = tmp / f"m{len(out)}{ext}"
    written = _copy_capped(src, dest, min(_MAX_MEMBER_BYTES, remaining))
    if written is None:
        if remaining < _MAX_MEMBER_BYTES:
            logger.warning(
                "aggregate extraction budget (%d bytes) reached for %s; stopping",
                _MAX_TOTAL_BYTES,
                archive_name,
            )
            return False
        logger.warning("archive member too large, skipping: %s!%s", archive_name, name)
        return True
    state["total"] += written
    out.append((name, dest))
    return True


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
    state = {"total": 0}
    try:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    with zf.open(info) as src:
                        if not _accept(
                            info.filename, src, tmp, supported_exts, out, state, path.name
                        ):
                            break
        elif _is_tar(path):
            with tarfile.open(path) as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    src = tf.extractfile(member)
                    if src is None:
                        continue
                    with src:
                        if not _accept(
                            member.name, src, tmp, supported_exts, out, state, path.name
                        ):
                            break
        else:  # single-file compressor
            opener = _SINGLE_OPENERS[path.suffix.lower()]
            inner = path.name[: -len(path.suffix)]  # strip the compression suffix
            with opener(path, "rb") as src:
                _accept(inner, src, tmp, supported_exts, out, state, path.name)
        yield out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
