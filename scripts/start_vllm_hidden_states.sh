#!/usr/bin/env bash
# Serve the verifier with target hidden-state extraction for data generation.
# Start this before generate_hidden_states.sh.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
exec "$VLLM_PYTHON" "$SPECULATORS_SRC/scripts/launch_vllm.py" "$ROOT/Qwen3-4B-Instruct" \
  --target-layer-ids 2 18 33 \
  -- --port "${PORT:-8000}" --gpu-memory-utilization "${GPU_MEM:-0.9}"
