#!/usr/bin/env bash
# Launch a disconnect-proof training run in tmux (run ON zulu):
#   scripts/train_zulu.sh configs/125m.yaml my-run [extra keith-llm train args]
# Reattach:  tmux attach -t my-run
# Progress:  tail -f checkpoints/my-run/train.log
#            tail -1 checkpoints/my-run/metrics.jsonl | jq
set -euo pipefail

CONFIG=${1:?usage: train_zulu.sh <config.yaml> <run-name> [extra args]}
RUN=${2:?usage: train_zulu.sh <config.yaml> <run-name> [extra args]}
shift 2

cd "$(dirname "$0")/.."
mkdir -p "checkpoints/$RUN"

tmux new-session -d -s "$RUN" \
    "source .venv/bin/activate && \
     keith-llm train --config '$CONFIG' --out-dir 'checkpoints/$RUN' $* \
       2>&1 | tee 'checkpoints/$RUN/train.log'"

echo "training started in tmux session '$RUN'"
echo "  attach:  tmux attach -t $RUN"
echo "  log:     tail -f checkpoints/$RUN/train.log"
