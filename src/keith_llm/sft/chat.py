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


def _conversation_prompt_ids(
    tok: KeithTokenizer, history: list[tuple[str, str]], message: str
) -> list[int]:
    """Build the prompt token ids for a (possibly multi-turn) conversation,
    matching the SFT training layout: ``[bos]`` + each completed turn
    (``prompt(q) + a + [eos]``) + ``prompt(message)`` for the model to continue."""
    ids = [tok.bos_id]
    for user, assistant in history:
        ids += tok.encode(build_prompt(user)) + tok.encode(assistant.strip()) + [tok.eos_id]
    ids += tok.encode(build_prompt(message))
    return ids


def chat_once(
    model: Transformer,
    tok: KeithTokenizer,
    message: str,
    history: list[tuple[str, str]] | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    generator: torch.Generator | None = None,
) -> str:
    """Generate one reply to ``message``. ``history`` is prior (user, assistant)
    turns, rendered ahead of the message in the trained multi-turn layout.
    Returns only the new response text."""
    prompt_ids = _conversation_prompt_ids(tok, history or [], message)
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


def chat_repl(
    model: Transformer, tok: KeithTokenizer, history_turns: int = 6, **gen_kwargs
) -> None:
    """Interactive read-eval loop over stdin. Carries multi-turn history (the
    last ``history_turns`` exchanges) so follow-ups have context; 'reset' clears
    it, 'exit'/'quit' or EOF ends the session."""
    print("chat ready — 'reset' clears history, 'exit' quits")
    history: list[tuple[str, str]] = []
    while True:
        try:
            message = input("you> ").strip()
        except EOFError:
            break
        if message in ("exit", "quit"):
            break
        if message == "reset":
            history = []
            print("(history cleared)")
            continue
        if not message:
            continue
        reply = chat_once(model, tok, message, history=history, **gen_kwargs)
        print("bot>", reply)
        history = (history + [(message, reply)])[-history_turns:]
