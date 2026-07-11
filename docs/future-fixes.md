# Future fixes and features

A prioritized roadmap of known issues, quality improvements, and planned
features for keith-llm — captured as the project has grown so nothing is lost.
Items are grouped by kind and ordered roughly by impact within each group.

Legend: **[fix]** something broken/suboptimal today · **[feature]** new
capability · effort is a rough T-shirt size.

---

## Known issues (fixes)

### 1. SFT instruction-following collapse — *highest impact* [fix]
**Symptom.** The SFT chat model (`keith-llm-sft-500m`) forces almost every
question into its most-memorized template ("The X has an Armor Class of Y"),
ignoring the actual question — e.g. *"what weapons does a kobold have?"* →
*"The Kobolds has an Armor Class of 12."* It also garbles or repeats in
multi-turn ollama sessions.

**Root cause.** Two things, both in the SFT stage (not the base model):
- The grounded generator only produced **7 rigid question types** (type / AC /
  HP / speed / CR / one ability / stat-block) — no "weapons", "actions",
  "describe X", etc. Out-of-template questions are out-of-distribution.
- Trained **3 epochs to loss 0.0016** = memorized the templates rather than
  learning to follow instructions. Data was skewed ~16,437 grounded : 16
  creative.

**Fix.** (a) Much richer question variety and natural phrasing; (b) balance —
cap grounded examples per source/type, upsample creative; (c) train **1 epoch
at lower LR**, stopping while loss is ~0.5–1.0 (never let it collapse toward 0);
(d) add a multi-turn-safe chat template (or document that the model is
single-turn and use one-shot / `/clear`). See also #5 (synthetic data) — the
real unlock for variety. **Effort: M.**

### 2. SFT trainer has no periodic checkpointing [fix]
`SFTTrainer` saves only at the end, so an OOM or crash mid-run loses all
progress (hit during the 500m SFT). Add interval checkpointing (like the
pretraining `Trainer`) and optional resume. **Effort: S.**

### 3. SFT OOMs at `--batch-size 8` on a 24 GB card [fix]
A batch of several long stat-block examples padded to `max_seq_len` blows past
24 GB. Workaround today: `--batch-size 4` + `PYTORCH_CUDA_ALLOC_CONF=
expandable_segments:True`. Proper fix: **length-bucketed batching** (group
similar-length examples so a short batch isn't padded to a long one), or a
token-budget batch sampler. **Effort: S–M.**

### 4. RoPE complex ops break `torch.compile` fusion [fix]
`apply_rotary` uses complex multiplication, which TorchInductor can't codegen
("does not support code generation for complex operators") — a graph break in
the hot path every layer, costing throughput on long runs. Rewrite RoPE with
the mathematically identical **real-valued** interleaved rotation
(`x0·cos − x1·sin`, `x0·sin + x1·cos`); same convention, GGUF export unchanged,
guarded by the existing RoPE/KV-parity tests. **Effort: S.**

---

## Data & quality improvements

### 5. Synthetic instruction data (distillation) [feature]
The programmatic SFT generator is limited to fixed templates. Use a larger
model (e.g. Claude) to generate **varied instruction/response pairs grounded in
the corpus and 5etools JSON** — the lever for real breadth (creative tasks,
diverse phrasings, "describe/compare/explain" questions). Needs an Anthropic
API key on the host; costs per generation. Directly addresses #1. **Effort: M.**

### 6. Broaden and grow the pretraining corpus [feature]
Grow ~105M → several-hundred-M tokens by mixing **curated general-domain text**
(Project Gutenberg literature — public domain; Wikipedia — CC-BY-SA; curated
story collections) with the existing TTRPG sources. Benefits: less base
overfitting (the 500m base's val loss 2.93 was *worse* than the 125m's 2.83 for
lack of data), better language fundamentals (which makes SFT work better), and
broader knowledge.

Guardrails: **curate** (avoid raw internet scrape — noisy, hurts more than
helps); **don't dilute the domain** (oversample/repeat TTRPG so the model stays
specialized); **tag general docs `<|system:generic|>`** (already supported).
The ingest pipeline (manifest, licenses, OCR, dedup, cache) already handles
this — mostly a fetch script + manifest entries. **Effort: M.**

### 7. Model-based PDF extraction for hard scans/layouts [feature]
For PDFs the pdfplumber + OCR path still handles poorly (complex multi-column,
heavy tables), add an optional model-based extractor — **docling** (MIT, strong
on tables/reading order) or `unstructured` (Apache) — as a higher tier behind
the existing quality-audit gate. **Effort: M.**

### 8. Retrieval-augmented generation (RAG) for grounded Q&A [feature]
For reliable *factual* answers, index the corpus (embed chunks + vector store),
retrieve relevant passages at query time, and have the (SFT) model answer from
that context rather than from a small model's shaky memory. Adds an embedding
model + index + a retrieve-then-answer serving command. Complements SFT.
**Effort: L.**

---

## Scaling (bigger models)

### 9. 1B+ parameter models [feature]
Only worthwhile once the corpus is in the **billions** of tokens (at ~500M
tokens, a 500m model is better-matched than 1B). Full fp32-AdamW training of 1B
needs ~16 GB for optimizer/params/grads alone — beyond comfortable on a single
24 GB 4090. Requires memory infrastructure the stack lacks: **gradient
checkpointing**, an **8-bit optimizer** (bitsandbytes), and/or **LoRA/QLoRA**
(parameter-efficient fine-tuning — also the cheapest path to fine-tuning large
bases). **Effort: L.**

---

## Model & architecture features

### 10. Grouped-query attention (GQA) [feature]
Cheaper inference/memory for the 350M/500M presets (fewer KV heads). The model
already parameterizes `n_kv_heads == n_heads`; generalize attention + KV cache +
GGUF export to `n_kv_heads < n_heads`. **Effort: M.**

### 11. Structured stat-block generation [feature]
Grammar-constrained sampling so the model emits valid, parseable stat blocks
(and other structured content) rather than free text. **Effort: M.**

### 12. Perplexity-based quant-quality gate [feature]
Compare exported GGUF quants (Q8_0 / Q5_K_M / Q4_K_M) against the f16 model by
perplexity, and gate/report the quality drop before shipping a quant.
**Effort: S–M.**

### 13. Corpus expansion: more open systems [feature]
Add Pathfinder 2e (ORC license) and more open adventures/settings, with matching
`system`/`doc_type` control tokens. **Effort: S** (data) per source.

### 14. Multi-turn-safe chat serving [fix/feature]
The current SFT model is single-turn; ollama's interactive mode accumulates
history and degrades. Either train on multi-turn conversations, or ship a
proper `{{ .Messages }}` chat template that resets per turn, plus document the
one-shot / `/clear` workaround. **Effort: S–M.**

---

*Have a new idea or hit a new bug? Add it here so the roadmap stays the single
source of truth.*
