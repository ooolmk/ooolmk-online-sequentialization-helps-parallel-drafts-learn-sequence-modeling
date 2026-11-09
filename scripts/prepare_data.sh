#!/usr/bin/env bash
# Tokenize the on-policy responses into the Arrow dataset used by training.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env.sh"

exec "$SPECULATORS_PYTHON" "$SPECULATORS_SRC/scripts/prepare_data.py" \
  --model "$ROOT/Qwen3-4B-Instruct" \
  --data "$ROOT/qwen3_4b_dflash_300k/on_policy.jsonl" \
  --output "$ROOT/qwen3_4b_dflash_300k/prepared" \
  --seq-length 8192
