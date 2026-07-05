"""Full-pipeline CPU integration test, driven through the real CLI:

raw files -> ingest -> tokenizer -> binarize -> train -> generate -> GGUF.
"""

import json
import shutil

import gguf as gguf_lib
import pytest
import yaml

from keith_llm import cli
from keith_llm.tokenizer.wrapper import KeithTokenizer

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def workspace(tmp_path_factory, make_pdf, corpus_records_factory):
    """Raw sources + manifest laid out like a real checkout."""
    root = tmp_path_factory.mktemp("e2e")
    docs = root / "raw" / "dnd5e"
    docs.mkdir(parents=True)
    records = corpus_records_factory(n_docs=24)
    for i, rec in enumerate(records[:-1]):
        (docs / f"doc{i}.txt").write_text(rec["text"])
    # One multi-page PDF goes through the pypdf path.
    pdf_pages = [records[-1]["text"][i : i + 400] for i in range(0, 1200, 400)]
    (docs / "book.pdf").write_bytes(
        make_pdf([p.replace("(", "").replace(")", "") for p in pdf_pages])
    )
    manifest = root / "sources.yaml"
    manifest.write_text(
        "sources:\n"
        "  - glob: 'raw/dnd5e/**/*'\n    system: dnd5e\n    doc_type: adventure\n"
        "    license: CC-BY-4.0\n    publishable: true\n"
    )
    return root


def test_full_pipeline(workspace, tmp_path_factory):
    root = workspace
    out = tmp_path_factory.mktemp("e2e_out")
    corpus = out / "corpus.jsonl"
    tok_path = out / "tokenizer.json"
    bins = out / "tokens"

    # 1. ingest
    assert (
        cli.main(
            [
                "ingest",
                "--manifest",
                str(root / "sources.yaml"),
                "--out",
                str(corpus),
                "--root",
                str(root),
            ]
        )
        == 0
    )
    records = [json.loads(ln) for ln in corpus.read_text().splitlines()]
    assert len(records) >= 20
    assert all(r["system"] == "dnd5e" and r["publishable"] for r in records)

    # 2. tokenizer
    assert (
        cli.main(
            [
                "train-tokenizer",
                "--corpus",
                str(corpus),
                "--out",
                str(tok_path),
                "--vocab-size",
                "512",
            ]
        )
        == 0
    )
    tok = KeithTokenizer.load(tok_path)

    # 3. binarize
    assert (
        cli.main(
            [
                "binarize",
                "--corpus",
                str(corpus),
                "--tokenizer",
                str(tok_path),
                "--out-dir",
                str(bins),
                "--val-mod",
                "3",
            ]
        )
        == 0
    )
    meta = json.loads((bins / "meta.json").read_text())
    assert meta["n_train_tokens"] > 0 and meta["n_val_tokens"] > 0

    # 4. train (30 steps, CPU)
    config = out / "e2e.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "vocab_size": tok.vocab_size,
                    "d_model": 64,
                    "n_layers": 2,
                    "n_heads": 2,
                    "ffn_hidden": 128,
                    "max_seq_len": 128,
                },
                "train": {
                    "batch_size": 8,
                    "max_steps": 30,
                    "warmup_steps": 3,
                    "lr": 3.0e-3,
                    "min_lr": 3.0e-4,
                    "eval_interval": 15,
                    "eval_batches": 2,
                    "checkpoint_interval": 15,
                    "data_dir": str(bins),
                    "out_dir": str(out / "run"),
                },
            }
        )
    )
    assert (
        cli.main(
            ["train", "--config", str(config), "--tokenizer", str(tok_path), "--device", "cpu"]
        )
        == 0
    )
    metrics = [json.loads(ln) for ln in (out / "run" / "metrics.jsonl").read_text().splitlines()]
    assert metrics[-1]["loss"] < metrics[0]["loss"], "training must reduce loss"

    # 5. conditioned generation
    assert (
        cli.main(
            [
                "generate",
                "--config",
                str(config),
                "--ckpt",
                str(out / "run" / "latest.pt"),
                "--tokenizer",
                str(tok_path),
                "--system",
                "dnd5e",
                "--doc-type",
                "adventure",
                "--prompt",
                "The village of",
                "--max-new-tokens",
                "40",
                "--seed",
                "1",
                "--device",
                "cpu",
            ]
        )
        == 0
    )

    # 6. GGUF export + read-back
    gguf_path = out / "model-f16.gguf"
    assert (
        cli.main(
            [
                "export",
                "--ckpt",
                str(out / "run" / "latest.pt"),
                "--tokenizer",
                str(tok_path),
                "--out",
                str(gguf_path),
                "--name",
                "keith-llm-e2e",
            ]
        )
        == 0
    )
    reader = gguf_lib.GGUFReader(gguf_path)
    assert reader.get_field("general.architecture").contents() == "llama"
    assert len(reader.get_field("tokenizer.ggml.tokens").contents()) == tok.vocab_size
    assert reader.get_field("llama.block_count").contents() == 2

    # 7. quantize when llama.cpp is available (exercised on the GPU host)
    if shutil.which("llama-quantize"):
        assert cli.main(["quantize", str(gguf_path), "Q8_0"]) == 0
