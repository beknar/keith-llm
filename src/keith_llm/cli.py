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

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
