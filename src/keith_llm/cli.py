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


def _cmd_audit_corpus(args: argparse.Namespace) -> int:
    from keith_llm.data.audit import audit_corpus

    report = audit_corpus(args.corpus)
    if args.out:
        from pathlib import Path

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")

    print(f"documents: {report['n_documents']}  verdicts: {report['verdicts']}")
    worst = [d for d in report["documents"] if d["verdict"] != "OK"][: args.top]
    if worst:
        print(f"\n{'verdict':<5} {'wordlike':>8} {'intCaps':>8} {'w/line':>7} {'alpha':>6}  source")
        for d in worst:
            print(
                f"{d['verdict']:<5} {d['wordlike_frac']:>8.2f} {d['internal_caps_rate']:>8.2f} "
                f"{d['words_per_line']:>7.1f} {d['alpha_ratio']:>6.2f}  {d['source']}"
            )
    else:
        print("all documents look clean")
    return 0


def _cmd_fetch_5etools(args: argparse.Namespace) -> int:
    from keith_llm.data.fivetools import fetch_all

    categories = tuple(args.categories.split(","))
    stats = fetch_all(args.base_url, args.out_dir, categories=categories)
    print(json.dumps(stats, indent=2))
    return 1 if stats["files"] == 0 else 0


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


def _cmd_train(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from keith_llm.config import load_config
    from keith_llm.train.loop import Trainer

    model_cfg, train_cfg = load_config(args.config)
    if args.data_dir:
        train_cfg = replace(train_cfg, data_dir=args.data_dir)
    if args.out_dir:
        train_cfg = replace(train_cfg, out_dir=args.out_dir)
    if args.max_steps:
        train_cfg = replace(train_cfg, max_steps=args.max_steps)
    trainer = Trainer(
        model_cfg,
        train_cfg,
        device=args.device,
        resume=args.resume,
        tokenizer_path=args.tokenizer,
    )
    final_loss = trainer.train()
    print(f"done: step={trainer.step} loss={final_loss:.4f} out={train_cfg.out_dir}")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    import torch

    from keith_llm.config import load_config
    from keith_llm.generate import generate
    from keith_llm.model import Transformer
    from keith_llm.tokenizer.wrapper import KeithTokenizer

    model_cfg, _ = load_config(args.config)
    tok = KeithTokenizer.load(args.tokenizer)
    if tok.vocab_size != model_cfg.vocab_size:
        raise SystemExit(
            f"tokenizer vocab ({tok.vocab_size}) != model vocab ({model_cfg.vocab_size}); "
            "use the config the checkpoint was trained with"
        )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(model_cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt.get("model_state", ckpt))

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
    prompt_ids = tok.control_prefix(args.system, args.doc_type)
    if args.prompt:
        prompt_ids += tok.encode(args.prompt)
    out_ids = generate(
        model,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        stop_ids=[tok.eos_id],
        generator=generator,
    )
    print(tok.decode(out_ids))
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from keith_llm.export.gguf_export import export_gguf

    out = export_gguf(args.ckpt, args.tokenizer, args.out, name=args.name)
    print(out)
    return 0


def _cmd_quantize(args: argparse.Namespace) -> int:
    from keith_llm.export.quantize import quantize

    out = quantize(args.gguf, args.qtype, out_path=args.out, bin_path=args.bin)
    print(out)
    return 0


def _cmd_ollama(args: argparse.Namespace) -> int:
    from keith_llm.export.ollama import register, write_modelfile

    modelfile = write_modelfile(args.gguf)
    register(args.name, modelfile)
    print(f"registered {args.name}; try: ollama run {args.name}")
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

    p = sub.add_parser(
        "audit-corpus",
        help="score corpus.jsonl documents for PDF-extraction quality problems",
    )
    p.add_argument("--corpus", default="data/processed/corpus.jsonl")
    p.add_argument("--out", default=None, help="write the full JSON report here")
    p.add_argument("--top", type=int, default=25, help="how many worst docs to print")
    p.set_defaults(func=_cmd_audit_corpus)

    p = sub.add_parser(
        "fetch-5etools",
        help="import a self-hosted 5etools mirror as raw text into data/raw/dnd5e",
    )
    p.add_argument("--base-url", required=True, help="e.g. http://192.168.1.64")
    p.add_argument("--out-dir", default="data/raw/dnd5e")
    p.add_argument("--categories", default="adventures,books,bestiary")
    p.set_defaults(func=_cmd_fetch_5etools)

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

    p = sub.add_parser("train", help="train a model from token bins")
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", default=None, help="override train.data_dir")
    p.add_argument("--out-dir", default=None, help="override train.out_dir")
    p.add_argument("--max-steps", type=int, default=None, help="override train.max_steps")
    p.add_argument("--resume", default=None, help="checkpoint to resume from")
    p.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    p.add_argument("--device", default=None)
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("generate", help="generate text from a trained checkpoint")
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    p.add_argument("--system", default="generic", help="rule system to condition on")
    p.add_argument("--doc-type", default="adventure", help="document type to condition on")
    p.add_argument("--prompt", default="")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.set_defaults(func=_cmd_generate)

    p = sub.add_parser("export", help="export a checkpoint to f16 GGUF (llama arch)")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="keith-llm")
    p.set_defaults(func=_cmd_export)

    p = sub.add_parser("quantize", help="quantize a GGUF via llama.cpp llama-quantize")
    p.add_argument("gguf")
    p.add_argument("qtype", choices=["Q8_0", "Q5_K_M", "Q4_K_M"])
    p.add_argument("--out", default=None)
    p.add_argument("--bin", default=None, help="path to llama-quantize (or $LLAMA_QUANTIZE)")
    p.set_defaults(func=_cmd_quantize)

    p = sub.add_parser("ollama", help="write a Modelfile and register the GGUF with ollama")
    p.add_argument("--gguf", required=True)
    p.add_argument("--name", default="keith-llm")
    p.set_defaults(func=_cmd_ollama)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
