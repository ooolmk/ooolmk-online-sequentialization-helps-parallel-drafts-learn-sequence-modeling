#!/usr/bin/env bash
# Serve the verifier for response regeneration (chat completions only).
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec "$VLLM_PYTHON" -m vllm.entrypoints.cli.main serve "$ROOT/Qwen3-4B-Instruct" \
  --host 127.0.0.1 --port "${PORT:-8000}"
