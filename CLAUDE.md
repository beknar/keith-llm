# keith-llm

Small-LLM (1M–500M parameter) tool stack, built from scratch in PyTorch, for
generating fantasy TTRPG adventures across rule systems (D&D 5e, Savage
Worlds, the d6 System) and for inventing new rule systems.

## Goal

Own every stage of a narrow-domain LLM: corpus ingestion → custom BPE
tokenizer → Llama-style transformer → CUDA training → generation → GGUF
export → quantization → serving via llama.cpp / ollama. Generation is
conditioned with CTRL-style control tokens (`<|system:dnd5e|>`,
`<|doc:adventure|>`, …) baked into the corpus at binarization time.

## Codebase map

```
configs/            YAML presets: tiny-1m (CPU/tests), 25m, 125m (primary), 350m, 500m
data/sources.yaml   corpus manifest (globs → system/doc_type/license/publishable)
src/keith_llm/
  config.py         ModelConfig/TrainConfig dataclasses + YAML loader
  cli.py            `keith-llm` subcommand dispatcher (lazy imports)
  data/             ingest (pdf/md/txt, archive-aware), clean, dedup, corpus, binarize → uint16 memmap bins
  tokenizer/        byte-level BPE training + KeithTokenizer wrapper (control tokens)
  model.py          single-file Llama-style transformer (RoPE/RMSNorm/SwiGLU, KV cache)
  generate.py       sampling: temperature/top-k/top-p/repetition-penalty
  train/            Trainer, memmap batch loader, cosine LR, checkpointing, JSONL metrics
  export/           GGUF (llama arch) writer, llama-quantize wrapper, ollama Modelfile
scripts/            seed-data fetch + zulu (GPU host) setup/train/artifact scripts
tests/              unit/, integration/ (CI), soak/ (pytest -m soak, leak detection)
```

## Conventions

- Dev machine is CPU-only; everything must run with `configs/tiny-1m.yaml` on CPU.
  Training runs on `storm@zulu` (RTX 4090) in `/genai/keith-llm`.
- RoPE uses the Meta/llama2.c interleaved-pair convention so GGUF export needs
  no weight permutation — do not switch to HF rotate-half style.
- Install with the CPU torch index locally/CI:
  `pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cpu`
- Tests: `pytest` (soak excluded by default); soak: `pytest -m soak`.
- Workflow: branch → PR → review → merge; CI (ruff + pytest) must be green.
