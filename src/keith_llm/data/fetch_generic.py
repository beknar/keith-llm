"""Fetch vetted generic (non-domain) text into the corpus tree.

Small domain corpora make small models overfit; mixing in general English prose
(encyclopedic + literary) helps them generalize. This downloads openly-licensed
corpora from Hugging Face over its CDN — **not** by scraping the source sites, so
there is no crawler and no IP-ban risk — and writes them as ``.txt`` shards under
``data/seed/generic/<name>/`` where ``keith-llm ingest`` already picks them up
(``data/sources.yaml`` ships matching entries: ``system: generic``).

The heavy ``datasets`` dependency is an optional extra; install with
``pip install -e ".[fetch]"``. The network read is isolated in ``_stream_texts``
so the writer/orchestration are unit-testable without a network.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Recipe:
    """A named, vetted Hugging Face dataset to harvest as generic prose."""

    repo_id: str
    config: str | None
    split: str
    text_column: str
    license: str
    doc_type: str  # must be one of constants.DOC_TYPES (control-token vocab is frozen)
    subdir: str
    note: str = ""


# Built-in vetted sources. All openly licensed; all tagged system=generic. The
# doc_type is "setting" (descriptive prose) because DOC_TYPES is frozen — the
# meaningful conditioning signal for this text is the <|system:generic|> token.
RECIPES: dict[str, Recipe] = {
    "wikitext": Recipe(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        "train",
        "text",
        "CC-BY-SA-4.0",
        "setting",
        "wikitext",
        "clean English Wikipedia prose (~500MB) — a good default diversity injection",
    ),
    "wikipedia": Recipe(
        "wikimedia/wikipedia",
        "20231101.en",
        "train",
        "text",
        "CC-BY-SA-4.0",
        "setting",
        "wikipedia",
        "full English Wikipedia (very large; bound it with --max-mb / --max-docs)",
    ),
    "gutenberg": Recipe(
        "manu/project_gutenberg",
        "default",
        "en",  # this dataset keys languages as splits, not configs
        "text",
        "public-domain",
        "setting",
        "gutenberg",
        "public-domain books (literary prose)",
    ),
}

# reader(repo_id, config, split, text_column) -> iterator of document strings
Reader = Callable[[str, str | None, str, str], Iterable[str]]


def _stream_texts(repo_id: str, config: str | None, split: str, text_column: str) -> Iterator[str]:
    """Stream a dataset's text column over the HF CDN. Streaming means we never
    download the whole corpus — the writer stops pulling once its caps are hit."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise SystemExit(
            "fetch-generic needs the 'fetch' extra: pip install -e \".[fetch]\""
        ) from exc

    ds = load_dataset(repo_id, config, split=split, streaming=True)
    for row in ds:
        text = row.get(text_column)
        if isinstance(text, str) and text.strip():
            yield text


def write_shards(
    texts: Iterable[str],
    out_dir: str | Path,
    name: str,
    *,
    shard_bytes: int = 4_000_000,
    max_docs: int | None = None,
    max_bytes: int | None = None,
    min_chars: int = 50,
) -> dict[str, Any]:
    """Consume ``texts`` and write them as numbered ``.txt`` shards under
    ``out_dir``, each up to ``shard_bytes``. Stops when ``max_docs`` documents or
    ``max_bytes`` total have been written (whichever first) — because the input is
    typically a lazy stream, stopping here stops the download. Returns counts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = docs = total = 0
    buf: list[str] = []
    buf_bytes = 0
    idx = 1

    def flush() -> None:
        nonlocal files, buf, buf_bytes, idx
        if not buf:
            return
        (out_dir / f"{name}-{idx:05d}.txt").write_text("\n\n".join(buf), encoding="utf-8")
        files += 1
        idx += 1
        buf = []
        buf_bytes = 0

    for text in texts:
        text = text.strip()
        if len(text) < min_chars:
            continue
        nbytes = len(text.encode("utf-8"))
        buf.append(text)
        buf_bytes += nbytes
        docs += 1
        total += nbytes
        if buf_bytes >= shard_bytes:
            flush()
        if max_docs is not None and docs >= max_docs:
            break
        if max_bytes is not None and total >= max_bytes:
            break
    flush()
    return {"docs": docs, "files": files, "bytes": total}


def fetch_one(
    recipe: Recipe,
    name: str,
    out: str | Path,
    *,
    max_docs: int | None = None,
    max_bytes: int | None = None,
    reader: Reader = _stream_texts,
) -> dict[str, Any]:
    """Harvest a single recipe into ``out/<subdir>/`` and return its stats."""
    dest = Path(out) / recipe.subdir
    texts = reader(recipe.repo_id, recipe.config, recipe.split, recipe.text_column)
    stats = write_shards(texts, dest, name, max_docs=max_docs, max_bytes=max_bytes)
    logger.info("fetched %s -> %s: %s", name, dest, stats)
    return {**stats, "path": str(dest), "license": recipe.license, "repo_id": recipe.repo_id}


def fetch_generic(
    sources: Iterable[str],
    out: str | Path = "data/seed/generic",
    *,
    max_docs: int | None = None,
    max_mb: float | None = 100.0,
    reader: Reader = _stream_texts,
) -> dict[str, Any]:
    """Harvest one or more built-in recipes (by name) into ``out``. ``max_mb`` and
    ``max_docs`` cap each source. Returns per-source stats."""
    max_bytes = int(max_mb * 1_000_000) if max_mb else None
    results: dict[str, Any] = {}
    for name in sources:
        recipe = RECIPES.get(name)
        if recipe is None:
            raise SystemExit(f"unknown source {name!r}; run 'keith-llm fetch-generic --list'")
        results[name] = fetch_one(
            recipe, name, out, max_docs=max_docs, max_bytes=max_bytes, reader=reader
        )
    return results
