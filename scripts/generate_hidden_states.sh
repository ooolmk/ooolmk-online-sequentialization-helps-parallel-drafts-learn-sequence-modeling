#!/usr/bin/env bash
# Precompute verifier hidden states for every training sample.
# Requires start_vllm_hidden_states.sh to be running on PORT.
# WARNING: the full 300k dataset produces ~13 TB on disk. Set MAX_SAMPLES for
# a smoke test, e.g. MAX_SAMPLES=2000.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env.sh"

ARGS=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  ARGS+=(--max-samples "$MAX_SAMPLES")
fi

exec "$SPECULATORS_PYTHON" "$SPECULATORS_SRC/scripts/data_generation_offline.py" \
  --model "$ROOT/Qwen3-4B-Instruct" \
  --preprocessed-data "$ROOT/qwen3_4b_dflash_300k/prepared" \
  --endpoint "http://127.0.0.1:${PORT:-8000}/v1" \
  --output "$ROOT/qwen3_4b_dflash_300k/hidden_states" \
  --concurrency "${CONCURRENCY:-64}" \
  --validate-outputs \
  --request-timeout 120 \
  --max-retries 3 \
  "${ARGS[@]}"
