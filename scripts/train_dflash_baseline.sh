#!/usr/bin/env bash
# Offline DFlash baseline used as the ckpt0-2 comparison reference.
# Produces qwen3_4b_dflash_300k/checkpoints/{0,1,2,3}.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec "$SPECULATORS_PYTHON" "$SPECULATORS_SRC/scripts/train.py" \
  --verifier-name-or-path "$ROOT/Qwen3-4B-Instruct" \
  --data-path "$ROOT/qwen3_4b_dflash_300k/prepared" \
  --hidden-states-path "$ROOT/qwen3_4b_dflash_300k/hidden_states" \
  --save-path "$ROOT/qwen3_4b_dflash_300k/checkpoints" \
  --speculator-type dflash \
  --target-layer-ids 2 18 33 \
  --num-layers 5 \
  --draft-vocab-size 151936 \
  --block-size 8 \
  --max-anchors 3072 \
  --epochs 4 \
  --lr 3e-4 \
  --scheduler-type cosine \
  --on-missing raise
