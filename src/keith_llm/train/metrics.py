"""Lightweight JSONL metrics logging — one JSON object per line, jq-friendly."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a")

    def log(self, **fields: Any) -> None:
        clean = {
            k: (round(v, 6) if isinstance(v, float) else v)
            for k, v in fields.items()
            if v is not None
        }
        self._fh.write(json.dumps(clean) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
