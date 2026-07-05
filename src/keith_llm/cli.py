"""Command-line entry point.

Subcommands are registered here but import their implementation lazily, so the
CLI stays fast to start and the package works even before every stage exists.
"""

from __future__ import annotations

import argparse
import sys

from keith_llm import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="keith-llm",
        description="Train and ship small LLMs for fantasy TTRPG adventure generation.",
    )
    parser.add_argument("--version", action="version", version=f"keith-llm {__version__}")
    parser.add_subparsers(dest="command")

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
