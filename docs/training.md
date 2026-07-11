# Training runbook (GPU host)

End-to-end commands for a full training + export cycle on the CUDA host
(`storm@zulu`, RTX 4090, project at `/genai/keith-llm`). Everything below
runs **on zulu** unless noted.

## 0. One-time setup

```bash
# from your workstation
scp scripts/setup_zulu.sh storm@zulu:/tmp/ && ssh storm@zulu bash /tmp/setup_zulu.sh
ssh storm@zulu /genai/keith-llm/scripts/setup_llamacpp.sh
```

Add to `~/.bashrc` on zulu:

```bash
export LLAMA_QUANTIZE=/genai/llama.cpp/build/bin/llama-quantize
export PATH=$PATH:/genai/llama.cpp/build/bin
```

Install ollama (optional, for serving): `curl -fsSL https://ollama.com/install.sh | sh`

## 1. Data

```bash
cd /genai/keith-llm && source .venv/bin/activate
git pull                      # code arrives via git, never rsync
scripts/fetch_seed_data.sh    # SRD 5.1 + OpenD6 into data/seed/
# Got a pile of messy source files (PDFs full of images, scans, HTML, JPGs,
# archives)? Pre-convert them to clean readable .txt first — image/scan OCR,
# garbled-text-layer re-OCR, clumped-word repair, and a gibberish gate that
# discards art/maps/no-text files (needs the `convert` extra + tesseract):
#   keith-llm convert --src /path/to/messy --out data/raw/<system>/<doc_type>
#
# drop your own books under data/raw/<system>/<doc_type>/ and add matching
# entries to data/sources.yaml. Archives (.zip, .tar[.gz/.bz2/.xz], .tgz) and
# single-file .gz/.bz2/.xz are expanded automatically; their contained
# .pdf/.txt/.md files are ingested. Unsupported files are logged and skipped.
#
# Have a pile of UNSORTED files? Let classify propose a doc type per file and
# sort them for you (it asks before moving; low-confidence files are left for
# you to sort by hand):
keith-llm classify --src data/raw/dnd5e/_inbox --system dnd5e   # review, confirm, done

keith-llm ingest              # -> data/processed/corpus.jsonl
```

PDFs are extracted column-aware via pdfplumber (with a pypdf fallback), which
fixes the multi-column reading-order corruption on rulebooks and modules.
**Scanned / image-only PDFs** (no text layer) are recovered with OCR when the
`ocr` extra and Tesseract are installed — otherwise they're dropped as low
quality. Enable it once on the training host:

```bash
sudo apt-get install -y tesseract-ocr        # system OCR engine
pip install -e ".[ocr]"                       # pypdfium2 (render) + pytesseract
```

Then `keith-llm ingest` OCRs only the image-only pages automatically (pages
with real text are untouched, so it's cheap on normal PDFs). OCR is slow on
fully-scanned books (~1-2 s/page); pass `keith-llm ingest --no-ocr` to skip it
for a quick run.

Extracted text is cached (content-addressed, in `data/cache/extraction.sqlite`),
so re-ingesting after adding a few files **skips re-extraction/OCR of every
unchanged file** — the second run is near-instant. The `cache_hits` stat shows
how many were served from cache. Use `--no-cache` to force a full re-extract;
delete `data/cache/` to reset it (it also rebuilds automatically if the
extraction logic or OCR availability changes).

Audit extraction quality before tokenizing — this scores each document and
lists the worst offenders (usually bad PDFs) so you can fix or drop them:

```bash
keith-llm audit-corpus --out data/processed/audit.json
# verdict BAD/WARN flags low wordlike-fraction or high internal-caps
# (column-interleave) documents; OK docs are omitted from the printout
```

Then eyeball a few flagged docs: `jq -r '.documents[] | select(.verdict!="OK")
| .source' data/processed/audit.json` and read them with
`shuf -n 3 data/processed/corpus.jsonl | jq -r .text | less`.

Check for the same content ingested from two sources (e.g. an adventure present
as both a 5etools text render and a PDF). This measures overlap by containment,
so it catches duplicates even when the two copies differ in size/format:

```bash
keith-llm dedup-report --out data/processed/dedup.json   # dry run: lists KEEP/DROP per cluster
# apply (keeps the cleaner copy, quarantines the rest to data/quarantine/):
keith-llm dedup-report --apply
# or delete permanently instead of quarantining:
keith-llm dedup-report --apply --hard
```

`--apply` removes the flagged **source files**, so re-run `keith-llm ingest`
afterward to rebuild the corpus without them. Quarantined files can be restored
from `data/quarantine/` (delete that directory to make removal permanent).

```bash
keith-llm train-tokenizer     # -> data/tokenizer/tokenizer.json (vocab 16384)
keith-llm binarize            # -> data/tokens/{train,val}.bin + meta.json
jq . data/tokens/meta.json    # sanity: token counts, vocab
```

Pick `max_steps` from the corpus size: `steps ≈ epochs × n_train_tokens /
(batch_size × grad_accum × max_seq_len)`. 15–25 epochs is reasonable for the
125m preset; watch for train/val divergence past that.

## 2. Train

```bash
scripts/train_zulu.sh configs/25m.yaml sanity-25m     # ~1-2h sanity run first
scripts/train_zulu.sh configs/125m.yaml prod-125m     # primary model
```

Monitor (any of):

```bash
tmux attach -t prod-125m                 # detach: Ctrl-b d
tail -f checkpoints/prod-125m/train.log
tail -1 checkpoints/prod-125m/metrics.jsonl | jq '{step,loss,val_loss,tok_per_sec}'
watch -n5 nvidia-smi
less checkpoints/prod-125m/samples.txt   # periodic conditioned samples
```

Interrupt/resume:

```bash
keith-llm train --config configs/125m.yaml --out-dir checkpoints/prod-125m \
    --resume checkpoints/prod-125m/latest.pt
```

## 3. Export, quantize, serve

```bash
mkdir -p exports
keith-llm export --ckpt checkpoints/prod-125m/latest.pt \
    --out exports/keith-llm-125m-f16.gguf --name keith-llm-125m
keith-llm quantize exports/keith-llm-125m-f16.gguf Q8_0
keith-llm quantize exports/keith-llm-125m-f16.gguf Q5_K_M
keith-llm quantize exports/keith-llm-125m-f16.gguf Q4_K_M
```

Q8_0 is the recommended quant at these model sizes; compare the others
against it before shipping.

Validate in llama.cpp directly (note: no `<|bos|>` in the prompt — the GGUF
sets `add_bos_token=true`; current llama.cpp uses `llama-completion` for raw
completion, `llama-cli` is chat-only):

```bash
llama-completion -m exports/keith-llm-125m-Q8_0.gguf --special \
    -p '<|system:dnd5e|><|doc:adventure|>The village of Emberfall' -n 200

# tokenizer parity: these two must print identical id lists
llama-tokenize -m exports/keith-llm-125m-Q8_0.gguf -p "some test text" --ids
python -c 'from keith_llm.tokenizer.wrapper import KeithTokenizer as K; \
    t=K.load("data/tokenizer/tokenizer.json"); print([t.bos_id]+t.encode("some test text"))'
```

Register and run with ollama:

```bash
keith-llm ollama --gguf exports/keith-llm-125m-Q8_0.gguf --name keith-llm-125m
ollama run keith-llm-125m '<|system:dnd5e|><|doc:adventure|>The village of Emberfall'
```

## 4. Soak tests (CUDA)

```bash
KEITH_SOAK_ITERS=20000 pytest -m soak tests/soak/test_soak_train.py -q
KEITH_SOAK_GEN_ITERS=5000 pytest -m soak tests/soak/test_soak_generate.py -q
```

Both fail on sustained RSS/CUDA-memory growth (slope regression after
warmup), not just absolute limits.

## 5. Pull artifacts back (from your workstation)

```bash
scripts/pull_artifacts.sh prod-125m
```

## Troubleshooting

- **Garbage output in llama.cpp but fine in `keith-llm generate`** → RoPE or
  tokenizer mismatch. Check `llama-cli --special` tokenization of a control
  prompt against `KeithTokenizer.encode`; re-read the RoPE note in
  `src/keith_llm/model.py`.
- **`llama-quantize` k-quant errors** → tensor rows not divisible by 256;
  only GPU presets are k-quant-safe (tiny-1m is Q8_0-only).
- **OOM on 500m** → drop `batch_size`, raise `grad_accum` (tokens/step
  constant), or shorten `max_seq_len`.
- **Control tokens not taking effect via ollama** → confirm the specials
  parse as single tokens with `llama-cli --special`; fallback is baking the
  prefix into the Modelfile TEMPLATE.
