#!/usr/bin/env bash
# DSpark online hybrid training: DFlash backbone + sequential Markov logit bias
# and confidence head. Same hybrid objective and pipeline as DFlash.
# NOTE: this release reports results for DFlash only; DSpark results are not
# included. The training method is provided for reuse.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env.sh"

RUN_ROOT="${RUN_ROOT:-$ROOT/qwen3_4b_dflash_300k/online_rl/runs/dspark_hybrid_ddp2}"
exec "$SPECULATORS_PYTHON" "$ROOT/qwen3_4b_dflash_300k/online_rl/run_dspark_epoch_pipeline.py" \
  --run-root "$RUN_ROOT" \
  --epochs 4 \
  --seed 1234 \
  --max-anchors 3072 \
  --verify-batch 64 \
  --chunk-blocks 256 \
  --verifier-label-backend tree \
  --online-loss log_eal \
  --log-eal-eps 1e-6 \
  --kl-weight 1.0 \
  --kl-temperature 1.0 \
  --markov-rank 256 \
  --markov-head-type vanilla \
  --enable-confidence-head \
  --confidence-head-with-markov \
  --confidence-head-alpha 1.0 \
  --lr 3e-4 \
  --muon-lr 3e-3 \
  --train-data-ratio 0.999 \
  --num-workers 12 \
  --checkpoint-freq 0.01 \
  --nproc-per-node 2 \
  --vllm-python "$VLLM_PYTHON" \
  --eval-tag dspark_hybrid \
  --eval-port 8001 \
  --eval-cuda-visible-devices "${EVAL_GPU:-1}" \
  --eval-gpu-memory-utilization 0.5
