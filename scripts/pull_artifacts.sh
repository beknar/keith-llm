#!/usr/bin/env bash
# Pull run artifacts from the training host back to this machine:
#   scripts/pull_artifacts.sh my-run
# Copies latest.pt, metrics.jsonl, samples.txt, train.log and any GGUFs in
# exports/ — not every intermediate checkpoint.
set -euo pipefail

RUN=${1:?usage: pull_artifacts.sh <run-name>}
HOST=${KEITH_TRAIN_HOST:-storm@zulu}
REMOTE_DIR=${KEITH_REMOTE_DIR:-/genai/keith-llm}

cd "$(dirname "$0")/.."
mkdir -p "checkpoints/$RUN" exports

rsync -avz --progress \
    --include='latest.pt' --include='metrics.jsonl' \
    --include='samples.txt' --include='train.log' --exclude='*' \
    "$HOST:$REMOTE_DIR/checkpoints/$RUN/" "checkpoints/$RUN/"

rsync -avz --progress "$HOST:$REMOTE_DIR/exports/" exports/ || true

echo "artifacts in checkpoints/$RUN/ and exports/"
