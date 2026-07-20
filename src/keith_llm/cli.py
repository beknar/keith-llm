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


def _cmd_convert(args: argparse.Namespace) -> int:
    from keith_llm.data.convert import convert_tree

    stats = convert_tree(
        args.src,
        args.out,
        enable_ocr=not args.no_ocr,
        do_reflow=not args.no_reflow,
        do_fix_spacing=not args.no_fix_spacing,
        min_chars=args.min_chars,
        force=args.force,
    )
    print(json.dumps(stats, indent=2))
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    from keith_llm.data.corpus import build_corpus

    stats = build_corpus(
        args.manifest,
        args.out,
        root=args.root,
        enable_ocr=not args.no_ocr,
        use_cache=not args.no_cache,
    )
    print(json.dumps(stats, indent=2))
    return 0


def _cmd_classify(args: argparse.Namespace) -> int:
    from keith_llm.constants import SYSTEMS
    from keith_llm.data.classify import apply_moves, classify_paths

    if args.system not in SYSTEMS:
        raise SystemExit(f"--system must be one of {SYSTEMS}")

    results = classify_paths(
        args.src,
        system=args.system,
        dest=args.dest,
        enable_ocr=not args.no_ocr,
        min_confidence=args.min_confidence,
    )
    if not results:
        print(f"no classifiable documents found under {args.src}")
        return 0

    confident = [r for r in results if r.confident]
    review = [r for r in results if not r.confident]
    print(f"classified {len(results)} files (system={args.system}):\n")
    for r in confident:
        print(f"  {r.doc_type:<9} {r.confidence:.0%}  {r.path}  ->  {r.target}")
    for r in review:
        best = r.doc_type or "none"
        print(f"  REVIEW    {r.confidence:.0%}  {r.path}  (best guess: {best}, too low to move)")

    if not confident:
        print("\nnothing confident enough to move; sort the REVIEW files by hand.")
        return 0
    if args.dry_run:
        print(f"\n(dry run) {len(confident)} files would be moved.")
        return 0

    proceed = args.yes
    if not proceed:
        try:
            answer = input(f"\nMove {len(confident)} files into {args.dest}/? [y/N] ")
        except EOFError:
            answer = "n"
        proceed = answer.strip().lower() in ("y", "yes")

    if not proceed:
        print("no changes made.")
        return 0
    res = apply_moves(confident)
    print(f"\nmoved {len(res['moved'])} files; skipped {len(res['skipped'])} (target existed).")
    for p in res["skipped"]:
        print(f"  skipped (name already at target): {p}")
    print("Re-run 'keith-llm ingest' to pick them up with their new doc types.")
    if review:
        print(f"{len(review)} low-confidence files left in place for manual review.")
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


def _cmd_dedup_report(args: argparse.Namespace) -> int:
    from pathlib import Path

    from keith_llm.data.dedup_report import apply_removals, report_corpus

    report = report_corpus(args.corpus, threshold=args.threshold)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")

    print(
        f"documents: {report['n_documents']}  duplicate clusters: {report['n_clusters']}  "
        f"files flagged: {report['n_drop_files']}  (overlap >= {args.threshold:.0%})"
    )
    for c in report["clusters"]:
        k = c["keep"]
        print(f"\nKEEP  {k['source']}  [{k['verdict']}, {k['n_chars']} chars]")
        for d in c["drop"]:
            print(
                f"  DROP  {d['source']}  overlap={d['overlap_with_keep']:.0%}  "
                f"[{d['verdict']}, {d['n_chars']} chars]"
            )

    if not report["drop_files"]:
        print("\nno duplicates above threshold")
        return 0
    if args.apply:
        res = apply_removals(report["drop_files"], root=args.root, hard=args.hard)
        where = "deleted" if args.hard else f"quarantined under {res['quarantine']}"
        print(f"\n{len(res['removed'])} files {where}.")
        if res["missing"]:
            print(f"  {len(res['missing'])} already gone: {res['missing'][:3]}")
        print("Re-run 'keith-llm ingest' to rebuild the corpus without them.")
    else:
        print(
            f"\n(dry run) {report['n_drop_files']} files would be removed. "
            "Re-run with --apply to quarantine them (reversible), or --apply --hard to delete."
        )
    return 0


def _cmd_sft(args: argparse.Namespace) -> int:
    from keith_llm.sft.trainer import SFTTrainer

    trainer = SFTTrainer(
        base_ckpt=args.base,
        data_jsonl=args.data,
        tokenizer_path=args.tokenizer,
        out_dir=args.out_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
    )
    final_loss = trainer.train()
    print(
        f"done: {len(trainer.examples)} examples, {trainer.total_steps} steps, "
        f"loss {final_loss:.4f} -> {args.out_dir}"
    )
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    import torch

    from keith_llm.config import ModelConfig
    from keith_llm.model import Transformer
    from keith_llm.sft.chat import chat_once, chat_repl
    from keith_llm.tokenizer.wrapper import KeithTokenizer
    from keith_llm.train.checkpoint import load_checkpoint

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.ckpt, map_location=device)
    cfg = ModelConfig(**ckpt["model_cfg"])
    model = Transformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    tok = KeithTokenizer.load(args.tokenizer)

    gen = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    if args.message:
        print(chat_once(model, tok, args.message, **gen))
    else:
        chat_repl(model, tok, **gen)
    return 0


def _cmd_sft_build(args: argparse.Namespace) -> int:
    from keith_llm.sft.build import build_sft_dataset

    stats = build_sft_dataset(
        args.out,
        base_url=args.base_url,
        max_per_source=args.max_per_source,
        seed=args.seed,
        generator=args.generator,
        model=args.model,
        ollama_url=args.ollama_url,
        pairs_per_item=args.pairs_per_item,
        from_corpus=args.from_corpus,
        corpus_docs_per_system=args.corpus_docs_per_system,
        corpus_pairs_per_doc=args.corpus_pairs_per_doc,
        multi_turn=args.multi_turn,
    )
    print(json.dumps(stats, indent=2))
    return 1 if stats["total"] == 0 else 0


def _cmd_clean_llm(args: argparse.Namespace) -> int:
    from keith_llm.data.llm_clean import clean_corpus

    verdicts = ("BAD",) if args.bad_only else ("BAD", "WARN")
    stats = clean_corpus(
        args.corpus,
        args.out,
        model=args.model,
        ollama_url=args.ollama_url,
        target_verdicts=verdicts,
        min_overlap=args.min_overlap,
        min_retain=args.min_retain,
        drop_failed=args.drop_failed,
        max_chars=args.max_chars,
        max_docs=args.max_docs,
        dry_run=args.dry_run,
    )
    printable = {k: v for k, v in stats.items() if k != "report"}
    print(json.dumps(printable, indent=2))
    for r in stats["report"]:
        if r["action"] != "keep" or r.get("reason") != "not_improved":
            flow = f"{r['old']}->{r.get('new', '?')}" if "new" in r else r["old"]
            print(f"  {r['action']:<7} {flow:<10} ov={r.get('overlap', '-'):<5} {r['source']}")
    if args.dry_run:
        print("\n(dry run) nothing written; drop --dry-run to apply.")
    return 0


def _cmd_fetch_generic(args: argparse.Namespace) -> int:
    from keith_llm.data.fetch_generic import RECIPES, Recipe, fetch_generic, fetch_one

    if args.list:
        for key, r in RECIPES.items():
            print(f"{key:12} {r.repo_id:28} [{r.license}]  {r.note}")
        return 0

    max_bytes = int(args.max_mb * 1_000_000) if args.max_mb else None
    if args.repo_id:
        name = args.name or args.repo_id.split("/")[-1]
        recipe = Recipe(
            args.repo_id, args.config, args.split, args.text_column, "unknown", args.doc_type, name
        )
        stats = {
            name: fetch_one(recipe, name, args.out, max_docs=args.max_docs, max_bytes=max_bytes)
        }
    elif args.sources:
        stats = fetch_generic(
            args.sources, args.out, max_docs=args.max_docs, max_mb=args.max_mb or None
        )
    else:
        raise SystemExit("give one or more sources (see --list) or --repo-id for a custom dataset")

    print(json.dumps(stats, indent=2))
    print(
        f"\nWrote generic text under {args.out}. The built-in sources are pre-registered in "
        "data/sources.yaml; run 'keith-llm ingest' to fold them into the corpus.\n"
        "(For a custom --repo-id or non-default --out, add a matching sources.yaml entry first.)"
    )
    # `datasets` streaming leaves native/async worker threads (fsspec/aiohttp)
    # that crash during interpreter finalization ("PyGILState_Release ... no
    # thread-state", core dump / exit 134) when we stop early at a size/doc cap.
    # The output is already written and flushed, so hard-exit to skip that
    # teardown and return a clean status (so `fetch-generic && ingest` works).
    _hard_exit_after_fetch()
    return 0  # unreachable, but keeps the signature honest


def _hard_exit_after_fetch() -> None:
    import os

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


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

    modelfile = write_modelfile(args.gguf, chat=args.chat)
    register(args.name, modelfile)
    kind = "chat" if args.chat else "completion"
    print(f"registered {args.name} ({kind}); try: ollama run {args.name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    from keith_llm.constants import DOC_TYPES

    parser = argparse.ArgumentParser(
        prog="keith-llm",
        description="Train and ship small LLMs for fantasy TTRPG adventure generation.",
    )
    parser.add_argument("--version", action="version", version=f"keith-llm {__version__}")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser(
        "convert",
        help="convert a tree of PDFs/HTML/images/archives into clean readable .txt files",
    )
    p.add_argument("--src", required=True, help="directory of source files to convert")
    p.add_argument("--out", required=True, help="directory to write converted .txt files to")
    p.add_argument("--no-ocr", action="store_true", help="skip OCR of images and scanned PDFs")
    p.add_argument("--no-reflow", action="store_true", help="keep original line breaks")
    p.add_argument("--no-fix-spacing", action="store_true", help="don't split run-together words")
    p.add_argument("--min-chars", type=int, default=200, help="discard results shorter than this")
    p.add_argument(
        "--force", action="store_true", help="reconvert files even if an up-to-date .txt exists"
    )
    p.set_defaults(func=_cmd_convert)

    p = sub.add_parser("ingest", help="build cleaned, deduplicated corpus.jsonl from the manifest")
    p.add_argument("--manifest", default="data/sources.yaml")
    p.add_argument("--out", default="data/processed/corpus.jsonl")
    p.add_argument("--root", default=".", help="directory manifest globs are relative to")
    p.add_argument(
        "--no-ocr", action="store_true", help="skip OCR of image-only PDF pages (faster)"
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="ignore the extraction cache and re-extract every file",
    )
    p.set_defaults(func=_cmd_ingest)

    p = sub.add_parser(
        "classify",
        help="suggest doc types for unsorted files and (after confirming) sort them",
    )
    p.add_argument("--src", required=True, help="file or directory of unsorted documents")
    p.add_argument("--system", default="dnd5e", help="rule system these files belong to")
    p.add_argument(
        "--dest", default="data/raw", help="base dir to sort into (<dest>/<system>/<doc_type>/)"
    )
    p.add_argument("--min-confidence", type=float, default=0.45, help="below this, flag for review")
    p.add_argument("--yes", "-y", action="store_true", help="apply moves without the prompt")
    p.add_argument("--dry-run", action="store_true", help="show classifications, never move")
    p.add_argument("--no-ocr", action="store_true", help="don't OCR scanned pages while sampling")
    p.set_defaults(func=_cmd_classify)

    p = sub.add_parser(
        "audit-corpus",
        help="score corpus.jsonl documents for PDF-extraction quality problems",
    )
    p.add_argument("--corpus", default="data/processed/corpus.jsonl")
    p.add_argument("--out", default=None, help="write the full JSON report here")
    p.add_argument("--top", type=int, default=25, help="how many worst docs to print")
    p.set_defaults(func=_cmd_audit_corpus)

    p = sub.add_parser(
        "clean-llm",
        help="repair audit-flagged corpus docs with a local LLM (grounded, verified)",
    )
    p.add_argument("--corpus", default="data/processed/corpus.jsonl")
    p.add_argument("--out", default="data/processed/corpus.cleaned.jsonl")
    p.add_argument("--model", default="gpt-oss", help="local ollama model to clean with")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--bad-only", action="store_true", help="only clean BAD docs (skip WARN)")
    p.add_argument(
        "--min-overlap", type=float, default=0.80, help="min forward overlap (anti-invention)"
    )
    p.add_argument(
        "--min-retain", type=float, default=0.60, help="min reverse overlap (anti-deletion)"
    )
    p.add_argument(
        "--drop-failed", action="store_true", help="drop BAD docs that cleaning couldn't fix"
    )
    p.add_argument("--max-chars", type=int, default=6000, help="per-request chunk size")
    p.add_argument("--max-docs", type=int, default=None, help="cap flagged docs processed (trial)")
    p.add_argument("--dry-run", action="store_true", help="report outcomes without writing output")
    p.set_defaults(func=_cmd_clean_llm)

    p = sub.add_parser(
        "dedup-report",
        help="report (and optionally remove) documents that overlap another source",
    )
    p.add_argument("--corpus", default="data/processed/corpus.jsonl")
    p.add_argument("--threshold", type=float, default=0.75, help="overlap coefficient cutoff")
    p.add_argument("--out", default=None, help="write the full JSON report here")
    p.add_argument("--root", default=".", help="base dir the corpus 'source' paths are relative to")
    p.add_argument("--apply", action="store_true", help="remove the flagged source files")
    p.add_argument(
        "--hard", action="store_true", help="with --apply, delete permanently instead of quarantine"
    )
    p.set_defaults(func=_cmd_dedup_report)

    p = sub.add_parser("chat", help="chat locally with an SFT checkpoint (no ollama needed)")
    p.add_argument("--ckpt", required=True, help="SFT checkpoint (latest.pt)")
    p.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    p.add_argument("--message", default=None, help="one-shot message (omit for interactive REPL)")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--device", default=None)
    p.set_defaults(func=_cmd_chat)

    p = sub.add_parser("sft", help="instruction-tune (SFT) a base checkpoint on an SFT dataset")
    p.add_argument("--base", required=True, help="base checkpoint to fine-tune (latest.pt)")
    p.add_argument("--data", default="data/sft/sft.jsonl")
    p.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    p.add_argument("--out-dir", default="checkpoints/sft")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=None)
    p.set_defaults(func=_cmd_sft)

    p = sub.add_parser(
        "sft-build",
        help="build the SFT instruction dataset (hand-written seed + grounded 5etools Q/A)",
    )
    p.add_argument("--out", default="data/sft/sft.jsonl")
    p.add_argument("--base-url", default=None, help="5etools mirror for grounded Q/A (optional)")
    p.add_argument("--max-per-source", type=int, default=None, help="cap Q/A per bestiary source")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--generator",
        choices=["programmatic", "ollama", "both"],
        default="programmatic",
        help="how to make grounded pairs: fixed templates, LLM-synthesized (varied), or both",
    )
    p.add_argument(
        "--model", default="gpt-oss", help="local ollama model for --generator ollama/both"
    )
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--pairs-per-item", type=int, default=5, help="synthesized pairs per monster")
    p.add_argument(
        "--from-corpus",
        default=None,
        help="ingested corpus.jsonl to synthesize multi-system pairs from (uses the local LLM)",
    )
    p.add_argument(
        "--corpus-docs-per-system",
        type=int,
        default=15,
        help="docs sampled per system for --from-corpus",
    )
    p.add_argument(
        "--corpus-pairs-per-doc", type=int, default=5, help="synthesized pairs per corpus doc"
    )
    p.add_argument(
        "--multi-turn",
        type=int,
        default=0,
        help="append N synthetic multi-turn conversations (teaches turn-independence)",
    )
    p.set_defaults(func=_cmd_sft_build)

    p = sub.add_parser(
        "fetch-5etools",
        help="import a self-hosted 5etools mirror as raw text into data/raw/dnd5e",
    )
    p.add_argument("--base-url", required=True, help="e.g. http://192.168.1.64")
    p.add_argument("--out-dir", default="data/raw/dnd5e")
    p.add_argument("--categories", default="adventures,books,bestiary")
    p.set_defaults(func=_cmd_fetch_5etools)

    p = sub.add_parser(
        "fetch-generic",
        help="download vetted generic (non-domain) text via HF CDN into the corpus tree",
    )
    p.add_argument("sources", nargs="*", help="built-in recipe names (see --list)")
    p.add_argument("--list", action="store_true", help="list built-in sources and exit")
    p.add_argument("--out", default="data/seed/generic", help="where to write .txt shards")
    p.add_argument("--max-docs", type=int, default=None, help="cap documents per source")
    p.add_argument(
        "--max-mb", type=float, default=100.0, help="cap MB written per source (0 = no cap)"
    )
    # custom-dataset escape hatch (for any HF dataset not in the built-in list)
    p.add_argument("--repo-id", default=None, help="fetch an arbitrary HF dataset instead")
    p.add_argument("--config", default=None, help="dataset config/subset name (with --repo-id)")
    p.add_argument("--split", default="train", help="dataset split (with --repo-id)")
    p.add_argument("--text-column", default="text", help="text column name (with --repo-id)")
    p.add_argument("--name", default=None, help="output subdir name (with --repo-id)")
    p.add_argument(
        "--doc-type", default="setting", choices=DOC_TYPES, help="doc_type tag (with --repo-id)"
    )
    p.set_defaults(func=_cmd_fetch_generic)

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
    p.add_argument(
        "--chat",
        action="store_true",
        help="instruction/chat Modelfile for an SFT model (default is raw completion)",
    )
    p.set_defaults(func=_cmd_ollama)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
