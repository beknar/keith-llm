"""Assemble the SFT instruction dataset.

Combines the hand-written creative seed set (packaged ``seed.jsonl``) with
grounded field-level Q/A generated from a 5etools mirror's bestiary JSON. Output
is a shuffled JSONL of ``{"instruction", "response", "source"}`` — the input to
SFT training. Deterministic given the same seed.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from keith_llm.data.fivetools import _get_json
from keith_llm.sft.monster_qa import monster_qa

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
    base_url: str, rng: random.Random, max_per_source: int | None
) -> list[dict[str, Any]]:
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
                pairs = monster_qa(mon, rng)
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


def build_sft_dataset(
    out_path: str | Path,
    base_url: str | None = None,
    seed_path: str | Path = _SEED_PATH,
    max_per_source: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Write the SFT JSONL. ``base_url`` adds grounded bestiary Q/A from a
    5etools mirror; omit it for the hand-written seed set only."""
    rng = random.Random(seed)
    examples = _load_seed(Path(seed_path))
    n_seed = len(examples)
    if base_url:
        examples.extend(_bestiary_pairs(base_url, rng, max_per_source))

    rng.shuffle(examples)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    stats = {
        "total": len(examples),
        "seed": n_seed,
        "grounded": len(examples) - n_seed,
    }
    logger.info("SFT dataset written to %s: %s", out_path, stats)
    return stats
