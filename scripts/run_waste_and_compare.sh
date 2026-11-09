#!/usr/bin/env bash
# Per-position offline-correct/wasted statistics on the 1,563 test prompts and
# regeneration of the ckpt0-2 comparison table.
# Requires the hybrid and offline checkpoints plus their acceptance results.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env.sh"

HYBRID_RUN="${HYBRID_RUN:-$ROOT/qwen3_4b_dflash_300k/online_rl/runs/dflash_hybrid_ddp2}"
OFFLINE_ROOT="${OFFLINE_ROOT:-$ROOT/qwen3_4b_dflash_300k/checkpoints}"

"$SPECULATORS_PYTHON" "$ROOT/qwen3_4b_dflash_300k/online_rl/test_wasted_8groups.py" \
  --epochs 0 1 2 \
  --device "${WASTE_DEVICE:-cuda:0}" \
  --hybrid-run-root "$HYBRID_RUN" \
  --offline-ckpt-root "$OFFLINE_ROOT"

"$SPECULATORS_PYTHON" "$ROOT/qwen3_4b_dflash_300k/online_rl/compare_dflash_ckpt0_2.py" \
  --epochs 0 1 2
