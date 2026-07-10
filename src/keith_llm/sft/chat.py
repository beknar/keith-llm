"""Local chat inference for an SFT model — no ollama/GGUF needed.

Wraps a message in the same Instruction/Response prompt the model was trained
on, generates a reply, stops at ``<|eos|>``, and returns just the response
(the prompt echo and any run-on into a fake next turn are trimmed).
"""

from __future__ import annotations

import torch

from keith_llm.generate import generate
from keith_llm.model import Transformer
from keith_llm.sft.format import STOP_SEQUENCES, build_prompt
from keith_llm.tokenizer.wrapper import KeithTokenizer


def chat_once(
    model: Transformer,
    tok: KeithTokenizer,
    message: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    generator: torch.Generator | None = None,
) -> str:
    """Generate one reply to ``message``. Returns only the response text."""
    prompt_ids = [tok.bos_id] + tok.encode(build_prompt(message))
    out = generate(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stop_ids=[tok.eos_id],
        generator=generator,
    )
    text = tok.decode(out[len(prompt_ids) :])
    for stop in STOP_SEQUENCES:  # trim a run-on that started a new turn
        cut = text.find(stop)
        if cut != -1:
            text = text[:cut]
    return text.strip()


def chat_repl(model: Transformer, tok: KeithTokenizer, **gen_kwargs) -> None:
    """Interactive read-eval loop over stdin. 'exit'/'quit' or EOF ends it."""
    print("chat ready — type 'exit' to quit")
    while True:
        try:
            message = input("you> ").strip()
        except EOFError:
            break
        if message in ("exit", "quit"):
            break
        if not message:
            continue
        print("bot>", chat_once(model, tok, message, **gen_kwargs))
