"""Command-line entry point.

Subcommands import their implementation lazily, so the CLI stays fast to
start and the package works even before every stage exists.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from keith_llm import __version__


def _cmd_ingest(args: argparse.Namespace) -> int:
    from keith_llm.data.corpus import build_corpus

    stats = build_corpus(args.manifest, args.out, root=args.root)
    print(json.dumps(stats, indent=2))
    return 0


def _cmd_train_tokenizer(args: argparse.Namespace) -> int:
    from keith_llm.tokenizer.train import train_bpe

    tok = train_bpe(args.corpus, args.out, vocab_size=args.vocab_size)
    print(f"vocab_size={tok.vocab_size} -> {args.out}")
    return 0


def _cmd_binarize(args: argparse.Namespace) -> int:
    from keith_llm.data.binarize import binarize

    meta = binarize(args.corpus, args.tokenizer, args.out_dir, val_mod=args.val_mod)
    print(json.dumps(meta, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="keith-llm",
        description="Train and ship small LLMs for fantasy TTRPG adventure generation.",
    )
    parser.add_argument("--version", action="version", version=f"keith-llm {__version__}")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("ingest", help="build cleaned, deduplicated corpus.jsonl from the manifest")
    p.add_argument("--manifest", default="data/sources.yaml")
    p.add_argument("--out", default="data/processed/corpus.jsonl")
    p.add_argument("--root", default=".", help="directory manifest globs are relative to")
    p.set_defaults(func=_cmd_ingest)

    p = sub.add_parser("train-tokenizer", help="train the byte-level BPE tokenizer")
    p.add_argument("--corpus", default="data/processed/corpus.jsonl")
    p.add_argument("--out", default="data/tokenizer/tokenizer.json")
    p.add_argument("--vocab-size", type=int, default=16384)
    p.set_defaults(func=_cmd_train_tokenizer)

    p = sub.add_parser("binarize", help="tokenize the corpus into uint16 train/val bins")
    p.add_argument("--corpus", default="data/processed/corpus.jsonl")
    p.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    p.add_argument("--out-dir", default="data/tokens")
    p.add_argument("--val-mod", type=int, default=50, help="1-in-N documents go to val")
    p.set_defaults(func=_cmd_binarize)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
