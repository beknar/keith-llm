#!/usr/bin/env bash
# One-shot setup of the training host (run ON zulu, as storm):
#   creates /genai/keith-llm, clones the repo, builds a venv with CUDA torch.
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/genai/keith-llm}
REPO=${REPO:-https://github.com/beknar/keith-llm.git}
CUDA_INDEX=${CUDA_INDEX:-https://download.pytorch.org/whl/cu126}

if [ ! -d "$PROJECT_DIR" ]; then
    sudo mkdir -p "$PROJECT_DIR"
    sudo chown "$(whoami)" "$PROJECT_DIR"
fi

if [ ! -d "$PROJECT_DIR/.git" ]; then
    git clone "$REPO" "$PROJECT_DIR"
fi
cd "$PROJECT_DIR"

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]" --extra-index-url "$CUDA_INDEX"

python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available — check driver / torch wheel"
print(f"OK: torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
EOF

echo "setup complete: $PROJECT_DIR"
