"""Minimal client for a local ollama LLM (e.g. gpt-oss).

Used to generate synthetic training data and to clean up problematic ingested
text with a locally-hosted model — no API key, no per-token cost. Talks to the
ollama HTTP API over stdlib urllib (no extra dependency).

This is distinct from ``export/ollama.py``, which registers *our* model with
ollama for serving; this calls *another* model to help build the corpus.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaClient:
    def __init__(
        self,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.7,
        timeout: int = 180,
        retries: int = 2,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.retries = retries

    def available(self) -> bool:
        """True if the ollama server responds (not that the model is pulled)."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/version", timeout=5):
                return True
        except Exception:  # noqa: BLE001 - unreachable server -> unavailable
            return False

    def chat(self, user: str, system: str | None = None) -> str:
        """One-shot chat completion. Raises after exhausting retries."""
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}
        ]
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": self.temperature},
            }
        ).encode()

        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(
                    f"{self.base_url}/api/chat",
                    data=body,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.load(resp)["message"]["content"]
            except Exception as exc:  # noqa: BLE001 - retry transient failures
                last_err = exc
                logger.warning(
                    "ollama chat attempt %d/%d failed: %s", attempt + 1, self.retries + 1, exc
                )
        raise RuntimeError(f"ollama chat failed after {self.retries + 1} attempts: {last_err}")
