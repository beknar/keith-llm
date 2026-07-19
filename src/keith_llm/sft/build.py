"""Assemble the SFT instruction dataset.

Combines up to three sources into a shuffled JSONL of
``{"instruction", "response", "source"}`` — the input to SFT training:

1. the hand-written creative/instructional seed set (packaged ``seed.jsonl``),
2. grounded field-level Q/A from a 5etools mirror's bestiary JSON, and
3. varied, grounded, multi-task instruction pairs synthesized from real
   ingested corpus documents balanced across every system (``from_corpus``).

Deterministic given the same seed for the programmatic bestiary generator; the
``ollama``/``both`` bestiary generators and the corpus generator call a local
LLM at temperature, so those pairs vary run to run.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from keith_llm.data.fivetools import _get_json
from keith_llm.llm import DEFAULT_BASE_URL, OllamaClient
from keith_llm.sft.monster_qa import monster_qa
from keith_llm.sft.synth import synth_corpus_pairs, synth_monster_pairs

logger = logging.getLogger(__name__)

_SEED_PATH = Path(__file__).parent / "seed.jsonl"


def _load_seed(seed_path: Path) -> list[dict[str, Any]]:
    out = []
    with open(seed_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out.append(
                {"instruction": rec["instruction"], "response": rec["response"], "source": "seed"}
            )
    return out


def _bestiary_pairs(
    base_url: str,
    gen_fn: Callable[[dict[str, Any]], list[tuple[str, str]]],
    max_per_source: int | None,
) -> list[dict[str, Any]]:
    """Fetch monsters from the mirror and turn each into instruction pairs via
    ``gen_fn`` (programmatic templates or LLM-synthesized)."""
    base = base_url.rstrip("/")
    out: list[dict[str, Any]] = []
    try:
        index = _get_json(f"{base}/data/bestiary/index.json")
    except Exception as exc:  # noqa: BLE001 - no mirror -> just skip grounded pairs
        logger.warning("could not read bestiary index from %s: %s", base, exc)
        return out
    for source, filename in index.items():
        try:
            monsters = _get_json(f"{base}/data/bestiary/{filename}").get("monster", [])
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the build
            logger.warning("skipping bestiary %s: %s", source, exc)
            continue
        count = 0
        for mon in monsters:
            try:
                pairs = gen_fn(mon)
            except Exception as exc:  # noqa: BLE001 - one malformed monster must not kill the build
                logger.warning("skipping monster %s: %s", mon.get("name", "?"), exc)
                continue
            for instruction, response in pairs:
                out.append(
                    {
                        "instruction": instruction,
                        "response": response,
                        "source": f"bestiary/{source}",
                    }
                )
                count += 1
                if max_per_source is not None and count >= max_per_source:
                    break
            if max_per_source is not None and count >= max_per_source:
                break
    return out


def _sample_corpus_docs(
    corpus_path: str | Path,
    docs_per_system: int,
    rng: random.Random,
    systems: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Reservoir-sample up to ``docs_per_system`` documents *per system* from a
    corpus.jsonl. Streamed, so a multi-GB corpus needn't fit in memory, and the
    per-system cap keeps the sample balanced instead of dominated by the biggest
    systems (dnd5e/generic)."""
    reservoirs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    with open(corpus_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):  # a valid-JSON but non-object line (null, [...], 1)
                continue
            system = rec.get("system", "generic")
            if systems is not None and system not in systems:
                continue
            counts[system] += 1
            res = reservoirs[system]
            if len(res) < docs_per_system:
                res.append(rec)
            else:  # replace with decreasing probability -> uniform sample
                j = rng.randint(0, counts[system] - 1)
                if j < docs_per_system:
                    res[j] = rec
    return [doc for res in reservoirs.values() for doc in res]


def _corpus_pairs(
    corpus_path: str | Path,
    client,
    docs_per_system: int,
    pairs_per_doc: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Sample real documents balanced across every system and synthesize varied,
    grounded, multi-task instruction pairs from each — tagged with the doc's
    system. This is what broadens SFT beyond 5e monster Q/A to all systems."""
    out: list[dict[str, Any]] = []
    for doc in _sample_corpus_docs(corpus_path, docs_per_system, rng):
        try:
            pairs = synth_corpus_pairs(client, doc, pairs_per_doc)
        except Exception as exc:  # noqa: BLE001 - one bad doc must not kill the build
            logger.warning("skipping corpus doc %s: %s", doc.get("source", "?"), exc)
            continue
        system = doc.get("system", "generic")
        for instruction, response in pairs:
            out.append(
                {"instruction": instruction, "response": response, "source": f"corpus/{system}"}
            )
    return out


def _make_generator(
    generator: str, rng: random.Random, client, pairs_per_item: int
) -> Callable[[dict[str, Any]], list[tuple[str, str]]]:
    """Choose how to turn a monster into instruction pairs: fixed templates,
    LLM-synthesized (varied, grounded), or both."""

    def programmatic(mon):
        return monster_qa(mon, rng)

    def synthesized(mon):
        return synth_monster_pairs(client, mon, pairs_per_item)

    if generator == "programmatic":
        return programmatic
    if generator == "ollama":
        return synthesized
    if generator == "both":
        return lambda mon: programmatic(mon) + synthesized(mon)
    raise ValueError(f"unknown generator {generator!r} (programmatic | ollama | both)")


def build_sft_dataset(
    out_path: str | Path,
    base_url: str | None = None,
    seed_path: str | Path = _SEED_PATH,
    max_per_source: int | None = None,
    seed: int = 0,
    generator: str = "programmatic",
    model: str = "gpt-oss",
    ollama_url: str = DEFAULT_BASE_URL,
    pairs_per_item: int = 5,
    from_corpus: str | Path | None = None,
    corpus_docs_per_system: int = 15,
    corpus_pairs_per_doc: int = 5,
) -> dict[str, Any]:
    """Write the SFT JSONL from three optional sources: the hand-written seed
    (always), grounded 5etools bestiary Q/A (``base_url``), and varied
    multi-system instruction pairs synthesized from real corpus documents
    (``from_corpus``, an ingested corpus.jsonl). ``generator`` chooses how
    bestiary pairs are made: ``programmatic`` templates, ``ollama``-synthesized,
    or ``both``. Corpus pairs always use the local LLM, balanced across systems."""
    rng = random.Random(seed)
    client = None
    needs_llm = generator in ("ollama", "both") or from_corpus is not None
    if needs_llm:
        client = OllamaClient(model=model, base_url=ollama_url)
        if not client.available():
            raise SystemExit(
                f"ollama not reachable at {ollama_url}; start it, use --generator programmatic, "
                "or drop --from-corpus"
            )

    examples = _load_seed(Path(seed_path))
    n_seed = len(examples)
    if base_url:
        gen_fn = _make_generator(generator, rng, client, pairs_per_item)
        examples.extend(_bestiary_pairs(base_url, gen_fn, max_per_source))
    n_after_bestiary = len(examples)
    if from_corpus is not None:
        examples.extend(
            _corpus_pairs(from_corpus, client, corpus_docs_per_system, corpus_pairs_per_doc, rng)
        )

    rng.shuffle(examples)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    stats = {
        "total": len(examples),
        "seed": n_seed,
        "bestiary": n_after_bestiary - n_seed,
        "corpus": len(examples) - n_after_bestiary,
        "generator": generator,
    }
    logger.info("SFT dataset written to %s: %s", out_path, stats)
    return stats
