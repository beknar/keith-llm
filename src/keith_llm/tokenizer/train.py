"""BPE tokenizer training.

Byte-level end-to-end (GPT-2 pre-tokenizer regex + ByteLevel decoder) — this
is required for GGUF export, where the vocabulary maps directly onto
llama.cpp's ``gpt2`` tokenizer implementation.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from keith_llm.tokenizer.wrapper import SPECIAL_TOKENS, KeithTokenizer

logger = logging.getLogger(__name__)


def _corpus_texts(corpus_jsonl: str | Path) -> Iterator[str]:
    with open(corpus_jsonl) as fh:
        for line in fh:
            yield json.loads(line)["text"]


def train_bpe(
    corpus_jsonl: str | Path,
    out_path: str | Path,
    vocab_size: int = 16384,
) -> KeithTokenizer:
    """Train a byte-level BPE tokenizer on the corpus and save tokenizer.json.

    ``vocab_size`` is the FINAL size including the special tokens, which are
    appended after training so they hold the top ids.
    """
    if vocab_size >= 2**16:
        raise ValueError("vocab_size must fit in uint16 token bins")
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size - len(SPECIAL_TOKENS),
        special_tokens=[],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(_corpus_texts(corpus_jsonl), trainer=trainer)
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    if tokenizer.get_vocab_size() > vocab_size:
        raise RuntimeError(
            f"trained vocab {tokenizer.get_vocab_size()} exceeds requested {vocab_size}"
        )
    wrapped = KeithTokenizer(tokenizer)
    wrapped.save(out_path)
    logger.info("tokenizer trained: vocab %d -> %s", wrapped.vocab_size, out_path)
    return wrapped
