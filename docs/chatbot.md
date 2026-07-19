# Chatbot runbook (SFT → serve)

How to turn a trained **base** model into a **conversational chatbot** and
serve it. Runs **on zulu** (`/genai/keith-llm`, RTX 4090) unless noted.

## Base vs. chat — read this first

The `keith-llm train` output is a **base (completion) model**: give it an
opening and it *continues* the text. It has the corpus knowledge but does not
follow instructions or hold a conversation.

A **chatbot** is that base model **supervised-fine-tuned (SFT)** on
instruction/response pairs. SFT teaches the *behavior* (answer questions, do
tasks); the knowledge already came from pretraining. This runbook covers the
SFT step and everything after it.

Prerequisites:

- a trained base checkpoint (e.g. `checkpoints/500m-blend/latest.pt`) and the
  **tokenizer it was trained with** (`data/tokenizer/tokenizer.json`) — the SFT
  and export steps must use that same tokenizer,
- `gpt-oss` pulled in ollama (`ollama pull gpt-oss`) for the data generator,
- optionally a 5etools mirror for grounded bestiary Q/A.

```bash
cd /genai/keith-llm && source .venv/bin/activate
git pull
```

## 1. Build the SFT dataset

`keith-llm sft-build` assembles instruction data from up to three sources: the
hand-written seed (always), grounded 5etools bestiary Q/A (`--base-url`), and
varied multi-system pairs synthesized from the ingested corpus (`--from-corpus`).

**Data variety is what makes or breaks the chatbot.** A dataset of a few rigid
templates causes *template collapse* — the model answers every prompt with the
same shape. Use `--generator both` (programmatic + LLM-varied) and
`--from-corpus` (real docs across every system) for a broad, varied set.

```bash
keith-llm sft-build --out data/sft/sft.jsonl \
    --base-url http://192.168.1.64 --generator both --model gpt-oss \
    --from-corpus data/processed/corpus.clean.jsonl \
    --corpus-docs-per-system 20 --max-per-source 40
```

- `--generator both` — programmatic templates + gpt-oss-synthesized (varied) monster Q/A.
- `--from-corpus` — samples `--corpus-docs-per-system` docs *per system* (balanced,
  so dnd5e/generic don't dominate) and synthesizes varied, grounded, multi-task
  pairs (Q&A, explain, describe, summarize, creative), tagged by system.
- Both LLM sources call gpt-oss once per item, so this is the **slow step**
  (~30–60 min). `--corpus-docs-per-system` and `--max-per-source` are the throttles.
- Runs quietly and writes the JSONL at the end; the final log line is a stats
  block (`seed` / `bestiary` / `corpus` / `total`).

Sanity-check variety before training (many distinct instruction stems = good;
collapse territory is a handful):

```bash
python -c 'import json,collections; \
  s=collections.Counter(" ".join(json.loads(l)["instruction"].split()[:4]) for l in open("data/sft/sft.jsonl")); \
  print(len(s),"distinct 4-word stems")'
```

## 2. SFT the base

> **Free the GPU first.** The data build leaves gpt-oss resident on the card
> (~13 GB). If you skip this, SFT will OOM.
>
> ```bash
> ollama stop gpt-oss
> ```

```bash
tmux new-session -d -s sft \
  "cd /genai/keith-llm && source .venv/bin/activate && \
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True keith-llm sft \
     --base checkpoints/500m-blend/latest.pt \
     --data data/sft/sft.jsonl \
     --tokenizer data/tokenizer/tokenizer.json \
     --out-dir checkpoints/500m-sft \
     --epochs 2 --batch-size 4 --lr 2e-5 \
   2>&1 | tee checkpoints/500m-sft/train.log"
```

Why these settings:

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and `--batch-size 4` — a
  500m base training/SFTs comfortably at batch 4; batch 8 has OOM'd on the 24 GB
  card. Set the env var **inside** the tmux command (it does not reliably
  propagate through `scripts/train_zulu.sh`).
- `--tokenizer` must be the one the base was trained with.
- **`--epochs 2` (light).** Over-training on SFT data drives loss toward zero and
  memorizes the templates — collapse. A **final loss around ~0.2 is healthy**;
  a loss near ~0.001 means it over-fit — reduce epochs and/or add data variety.

Progress: `tail -f checkpoints/500m-sft/train.log` (ends with a `done:` line).

## 3. Test it (before exporting)

```bash
keith-llm chat --ckpt checkpoints/500m-sft/latest.pt \
    --tokenizer data/tokenizer/tokenizer.json \
    --message "Give me an idea for a short dungeon adventure."
```

Ask several *different* questions. The chatbot is working if it gives distinct,
appropriate answers per prompt (numbers for stat questions, prose for creative
ones). If every answer looks the same → collapse; go back to step 1 (more
variety) and step 2 (fewer epochs).

## 4. Export → quantize → register

Use **distinct names** so a new chatbot doesn't overwrite an old one — the GGUF
filename is whatever you pass to `--out`, and the ollama model is `--name`;
nothing is auto-versioned.

```bash
mkdir -p exports
keith-llm export --ckpt checkpoints/500m-sft/latest.pt \
    --tokenizer data/tokenizer/tokenizer.json \
    --out exports/keith-500m-chat-f16.gguf --name keith-500m-chat

keith-llm quantize exports/keith-500m-chat-f16.gguf Q8_0 \
    --bin /genai/llama.cpp/build/bin/llama-quantize   # or set $LLAMA_QUANTIZE

# --chat writes an instruction Modelfile (### Instruction/### Response template).
# WITHOUT --chat you get a raw-completion Modelfile instead.
keith-llm ollama --gguf exports/keith-500m-chat-Q8_0.gguf \
    --name keith-500m-chat --chat
```

Q8_0 is the recommended quant at these sizes (a 500m Q8_0 is ~540 MB).

## 5. Run it

```bash
ollama run keith-500m-chat "Describe a spooky tavern for my players."
# or interactively:
ollama run keith-500m-chat
```

Or without ollama, straight from the checkpoint:

```bash
keith-llm chat --ckpt checkpoints/500m-sft/latest.pt \
    --tokenizer data/tokenizer/tokenizer.json     # REPL; omit --message
```

Tips:

- **Lead with creative/generative prompts** ("describe…", "give me an adventure
  hook for…", "invent a villain who…") — that's where a small model shines.
- Treat precise rules/stat lookups with skepticism; a 500m gets the *format*
  right but not always the exact numbers.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `CUDA out of memory` at SFT start | gpt-oss still on the GPU — `ollama stop gpt-oss`; keep `--batch-size 4` + `expandable_segments`. |
| Every answer is the same template | Template collapse — rebuild SFT data with more variety (`--generator both`, `--from-corpus`) and train fewer epochs. |
| SFT final loss ≈ 0.001 | Over-fit — reduce `--epochs`, add data. Healthy is ~0.2. |
| `ollama not reachable` from `sft-build` | Start ollama / point `--ollama-url` at the host; or drop the LLM sources. |
| New model replaced the old one | You reused `--out`/`--name` — use distinct names to keep both. |
| Chatbot ignores instructions, just rambles | That's the **base** model — you exported without SFT (or without `--chat`). |
