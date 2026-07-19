# keith-llm

A complete, from-scratch small-LLM tool stack for the fantasy tabletop RPG
domain. keith-llm trains 1M–500M parameter decoder-only language models on
TTRPG adventures and rule systems — D&D 5e, Savage Worlds, and the d6 System —
and ships them as quantized GGUF models that run in llama.cpp and ollama.

The project deliberately builds every stage itself rather than leaning on a
training framework: data ingestion, a custom BPE tokenizer, a Llama-style
transformer, the CUDA training loop, sampling, GGUF export, and quantization
are all first-party code. The result is a stack small enough to read
end-to-end and a model small enough to train on one RTX 4090.

## What it does

- **Generates fantasy TTRPG adventures** conditioned on a target rule system:
  ask for a D&D 5e dungeon crawl or a Savage Worlds pointcrawl and the model
  is steered by control tokens (`<|system:dnd5e|>`, `<|doc:adventure|>`) it
  learned during pretraining.
- **Drafts new rule systems**: a `<|system:homebrew|>` control token trained
  on homebrew/novel rules content switches the model into invention mode.

## Technologies

| Stage | Technology |
|---|---|
| Model & training | Pure PyTorch (bf16 autocast, fused AdamW, `torch.compile`) |
| Architecture | Llama-style: RoPE, RMSNorm, SwiGLU, tied embeddings, KV cache |
| Tokenizer | Byte-level BPE via HuggingFace `tokenizers`, 16,384 vocab |
| Data pipeline | pypdf + ftfy ingestion, MinHash dedup, `np.memmap` token bins |
| Export | `gguf` (llama architecture), llama.cpp `llama-quantize` |
| Serving | llama.cpp / ollama (Q8_0, Q5_K_M, Q4_K_M) |
| Testing | pytest: unit, integration (CI), and soak/leak-detection suites |

## Highlights

- **Reads like a paper implementation**: the entire transformer lives in one
  ~450-line `model.py`, nanoGPT-style.
- **GGUF export with no conversion scripts**: the model uses the
  Meta-convention interleaved RoPE that llama.cpp expects natively, so
  checkpoints export straight to `llama`-architecture GGUF — tokenizer
  embedded — and load in ollama unmodified.
- **License-aware corpus**: every document carries system/doc-type/license
  metadata; the openly licensed subset (CC-BY-4.0 SRD 5.1, OGL OpenD6) is
  separable from user-owned material at any time.
- **Leak-hunted**: dedicated soak tests regress RSS and CUDA memory over long
  train/generate runs and fail on upward slopes, not just absolute blowups.

## Documentation

- [CLAUDE.md](CLAUDE.md) — project summary and codebase map
- [docs/training.md](docs/training.md) — GPU training runbook (setup → train → quantize → ollama)
- [docs/chatbot.md](docs/chatbot.md) — chatbot runbook (SFT a base model → serve a conversational model)
- [data/sources.yaml](data/sources.yaml) — corpus manifest format

## Requirements

- Python 3.12+
- Local development: any Linux/WSL2 box (CPU is fine — tests and the
  `tiny-1m` preset never need a GPU)
- Training: an NVIDIA CUDA GPU (developed against an RTX 4090, 24 GB)
- Export/serving: llama.cpp (for quantization) and/or ollama
- Optional (scanned PDFs): the `ocr` extra plus the system `tesseract` binary

## Getting started

```bash
git clone https://github.com/beknar/keith-llm.git
cd keith-llm
python3 -m venv .venv && source .venv/bin/activate
# CPU machine (dev):
pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cpu
# CUDA machine (training):
pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu126
```

Then the pipeline, end to end:

```bash
keith-llm ingest                                  # corpus.jsonl from data/sources.yaml
keith-llm train-tokenizer                         # byte-level BPE → data/tokenizer/
keith-llm binarize                                # uint16 train.bin / val.bin
keith-llm train --config configs/125m.yaml        # (on the GPU host)
keith-llm generate --config configs/125m.yaml --ckpt checkpoints/125m/latest.pt \
    --system dnd5e --doc-type adventure --prompt "The village of Emberfall"
keith-llm export --ckpt checkpoints/125m/latest.pt --out exports/keith-llm-125m-f16.gguf
keith-llm quantize exports/keith-llm-125m-f16.gguf Q8_0
keith-llm ollama --gguf exports/keith-llm-125m-Q8_0.gguf --name keith-llm-125m
ollama run keith-llm-125m
```

On the GPU host, `scripts/train_zulu.sh configs/125m.yaml my-run` wraps the
train step in tmux so it survives disconnects — see
[docs/training.md](docs/training.md) for the full runbook.

## Running tests

```bash
pytest                 # unit + integration (what CI runs)
pytest -m integration  # just the end-to-end CPU pipeline test
pytest -m soak         # long-running memory-leak detection (not run in CI)
ruff check . && ruff format --check .
```

## Project structure

```
keith-llm/
├── configs/                 # model+training YAML presets (tiny-1m → 500m)
├── data/                    # sources.yaml manifest; corpora live here untracked
├── src/keith_llm/
│   ├── config.py            # dataclass configs + loader
│   ├── cli.py               # keith-llm entry point
│   ├── data/                # ingest → clean → dedup → corpus → binarize
│   ├── tokenizer/           # BPE training + wrapper with control tokens
│   ├── model.py             # Llama-style transformer
│   ├── generate.py          # sampling / inference
│   ├── train/               # training loop, checkpointing, metrics
│   └── export/              # GGUF export, quantization, ollama registration
├── scripts/                 # seed-data fetch, GPU-host setup/train/pull
├── tests/                   # unit/, integration/, soak/
└── docs/                    # runbooks
```

## Status: first training run (July 2026)

The full pipeline has been exercised end-to-end on an RTX 4090:

- Seed corpus: D&D 5.1 SRD (CC-BY-4.0) + seven OpenD6/OGL books → 8 documents,
  ~4.4M characters, 1.09M train / 117k val tokens (vocab 16,384)
- `25m` preset (~29M params), 600 steps @ ~450k tokens/sec bf16+compile,
  train loss 6.2 → 1.58 (the corpus is memorization-scale for now — grow
  `data/raw/` before trusting val loss)
- Exported `keith-llm-25m` GGUFs (f16/Q8_0/Q5_K_M/Q4_K_M); llama.cpp
  tokenization verified **byte-identical** to the Python tokenizer; control
  tokens parse as single specials; served via `ollama run keith-llm-25m`
- CUDA soak tests passed on the 4090: 10,000 training steps and 2,000
  generation cycles with flat RSS and CUDA-allocated memory
- The 125m+ presets are wired up and waiting on a larger corpus

## Future Fixes and Features

The prioritized roadmap of known issues, quality improvements, and planned
features lives in **[docs/future-fixes.md](docs/future-fixes.md)**. Highlights:

- **[fix]** SFT instruction-following collapse — richer/varied instruction data
  + lighter fine-tuning (the biggest lever on chatbot quality)
- **[feature]** Synthetic instruction data (distillation) for genuine breadth
- **[feature]** Broaden + grow the corpus with curated general-domain text
  (domain-upweighted, `generic`-tagged)
- **[feature]** Scaling to 1B+ (needs gradient checkpointing / 8-bit optimizer /
  LoRA), RAG for grounded Q&A, grouped-query attention, structured stat-block
  generation, and a perplexity-based quant-quality gate

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE).

Training data licensing is tracked separately per source in
[data/sources.yaml](data/sources.yaml); only openly licensed content
(CC-BY-4.0, OGL) is ever redistributed.
