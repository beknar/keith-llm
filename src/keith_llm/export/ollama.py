"""Ollama Modelfile generation and model registration."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from keith_llm.sft.format import INSTRUCTION_HEADER, RESPONSE_HEADER, STOP_SEQUENCES

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


def _chat_modelfile(gguf_name: str, num_predict: int) -> str:
    """Instruction/Response chat Modelfile for an SFT model. The TEMPLATE ranges
    over the whole conversation and renders each turn in the exact SFT layout
    (``build_prompt(user)`` + assistant + ``<|eos|>``) so a served *multi-turn*
    context byte-matches training — otherwise ollama would drop prior assistant
    turns and the model would see a malformed history. llama.cpp prepends
    <|bos|>; generation stops on <|eos|> and the instruction header."""
    template = "".join(
        [
            "{{- range .Messages }}",
            '{{- if eq .Role "user" }}',
            INSTRUCTION_HEADER,
            "{{ .Content }}",
            RESPONSE_HEADER,
            '{{ else if eq .Role "assistant" }}{{ .Content }}<|eos|>',
            "{{ end }}",
            "{{- end }}",
        ]
    )
    lines = [
        f"FROM ./{gguf_name}",
        f'TEMPLATE """{template}"""',
        'PARAMETER stop "<|eos|>"',
    ]
    lines += [f'PARAMETER stop "{s}"' for s in STOP_SEQUENCES]
    lines += [
        f"PARAMETER num_predict {num_predict}",
        "PARAMETER temperature 0.7",
        "PARAMETER top_p 0.95",
        "PARAMETER repeat_penalty 1.1",
    ]
    return "\n".join(lines) + "\n"


def write_modelfile(
    gguf_path: str | Path,
    out_path: str | Path | None = None,
    chat: bool = False,
    num_predict: int = 512,
) -> Path:
    """Write a Modelfile. ``chat=True`` wraps input in the SFT Instruction/
    Response template (for an instruction-tuned model); otherwise a raw-
    completion template (for a base model)."""
    gguf_path = Path(gguf_path)
    if not gguf_path.exists():
        raise FileNotFoundError(gguf_path)
    out_path = Path(out_path) if out_path else gguf_path.parent / "Modelfile"
    if chat:
        out_path.write_text(_chat_modelfile(gguf_path.name, num_predict))
    else:
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
