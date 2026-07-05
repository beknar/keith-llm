#!/usr/bin/env bash
# Build llama.cpp on the training host for llama-quantize and llama-cli.
# Run ON zulu. CPU build is enough for quantization and tokenizer parity
# checks; pass -DGGML_CUDA=ON via CMAKE_FLAGS for GPU inference.
set -euo pipefail

LLAMACPP_DIR=${LLAMACPP_DIR:-/genai/llama.cpp}
CMAKE_FLAGS=${CMAKE_FLAGS:-}

if [ ! -d "$LLAMACPP_DIR" ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMACPP_DIR"
fi
cd "$LLAMACPP_DIR"
git pull --ff-only || true

cmake -B build $CMAKE_FLAGS
cmake --build build -j"$(nproc)" --target llama-quantize llama-cli

BIN_DIR="$LLAMACPP_DIR/build/bin"
echo
echo "built: $BIN_DIR/llama-quantize, $BIN_DIR/llama-cli"
echo "add to your shell profile:"
echo "  export LLAMA_QUANTIZE=$BIN_DIR/llama-quantize"
echo "  export PATH=\$PATH:$BIN_DIR"
