"""Runtime tokenizer wrapper with the project's control-token scheme.

Conditioning is CTRL-style: every document is binarized with the prefix
``[<|bos|>, <|system:X|>, <|doc:Y|>]``, and generation steers output by
prepending the same prefix. Special tokens are added AFTER BPE training, so
they always occupy the top ids of the vocabulary.
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer

from keith_llm.constants import DOC_TYPES, SYSTEMS

BOS, EOS, PAD = "<|bos|>", "<|eos|>", "<|pad|>"
SPECIAL_TOKENS = (
    [BOS, EOS, PAD] + [f"<|system:{s}|>" for s in SYSTEMS] + [f"<|doc:{d}|>" for d in DOC_TYPES]
)


class KeithTokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self._tok = tokenizer
        missing = [t for t in SPECIAL_TOKENS if tokenizer.token_to_id(t) is None]
        if missing:
            raise ValueError(f"tokenizer missing special tokens: {missing}")

    @classmethod
    def load(cls, path: str | Path) -> KeithTokenizer:
        return cls(Tokenizer.from_file(str(path)))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def token_id(self, token: str) -> int:
        tid = self._tok.token_to_id(token)
        if tid is None:
            raise KeyError(token)
        return tid

    @property
    def bos_id(self) -> int:
        return self.token_id(BOS)

    @property
    def eos_id(self) -> int:
        return self.token_id(EOS)

    @property
    def pad_id(self) -> int:
        return self.token_id(PAD)

    def encode(self, text: str) -> list[int]:
        """Encode text. Special tokens embedded in the text are matched
        atomically (added tokens are never split by BPE)."""
        return self._tok.encode(text).ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special)

    def control_prefix(self, system: str, doc_type: str) -> list[int]:
        """``[bos, <|system:X|>, <|doc:Y|>]`` — the conditioning prefix used
        at both binarization and generation time."""
        if system not in SYSTEMS:
            raise ValueError(f"unknown system {system!r}, expected one of {SYSTEMS}")
        if doc_type not in DOC_TYPES:
            raise ValueError(f"unknown doc_type {doc_type!r}, expected one of {DOC_TYPES}")
        return [
            self.bos_id,
            self.token_id(f"<|system:{system}|>"),
            self.token_id(f"<|doc:{doc_type}|>"),
        ]
