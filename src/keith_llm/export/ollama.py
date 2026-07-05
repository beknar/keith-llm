"""Ollama Modelfile generation and model registration."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Raw-completion template: the base model has no chat format. Prompts should
# start with control tokens (e.g. "<|system:dnd5e|><|doc:adventure|>...");
# llama.cpp prepends <|bos|> itself (add_bos_token=true in the GGUF).
MODELFILE_TEMPLATE = """FROM ./{gguf_name}
TEMPLATE "{{{{ .Prompt }}}}"
PARAMETER stop <|eos|>
PARAMETER temperature 0.8
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1
"""


def write_modelfile(gguf_path: str | Path, out_path: str | Path | None = None) -> Path:
    gguf_path = Path(gguf_path)
    if not gguf_path.exists():
        raise FileNotFoundError(gguf_path)
    out_path = Path(out_path) if out_path else gguf_path.parent / "Modelfile"
    out_path.write_text(MODELFILE_TEMPLATE.format(gguf_name=gguf_path.name))
    return out_path


def register(name: str, modelfile_path: str | Path) -> None:
    """Run ``ollama create`` so the model shows up in ``ollama run``."""
    if shutil.which("ollama") is None:
        raise FileNotFoundError("ollama binary not found on PATH")
    subprocess.run(
        ["ollama", "create", name, "-f", str(Path(modelfile_path).resolve())],
        check=True,
        cwd=Path(modelfile_path).resolve().parent,
    )
    logger.info("registered ollama model %r", name)
