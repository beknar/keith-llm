"""The instruction/response text format shared by SFT training and serving.

Plain-text role markers (no new tokenizer tokens): the base model's existing
vocabulary tokenizes these normally, and SFT teaches the model the pattern.
Training and serving MUST use byte-identical headers, so both sides import
these constants — the ollama chat template is built from ``build_prompt`` too.

A training example is ``<|bos|>`` + prompt + response + ``<|eos|>``; loss is
computed only over the response and the final ``<|eos|>`` (see the SFT loader),
so the model learns to *produce* answers and to stop.
"""

from __future__ import annotations

INSTRUCTION_HEADER = "### Instruction:\n"
RESPONSE_HEADER = "\n\n### Response:\n"
# A generated turn ends at <|eos|>; the instruction header also serves as a
# stop sequence so a run-on can't start a fake next turn.
STOP_SEQUENCES = ("### Instruction:",)


def build_prompt(instruction: str) -> str:
    """The text fed to the model; generation continues after RESPONSE_HEADER."""
    return f"{INSTRUCTION_HEADER}{instruction.strip()}{RESPONSE_HEADER}"


def build_example_text(instruction: str, response: str) -> str:
    """Full training text (prompt + response), before bos/eos are added."""
    return build_prompt(instruction) + response.strip()
