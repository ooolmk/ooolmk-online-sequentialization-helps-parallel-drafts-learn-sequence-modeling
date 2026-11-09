#!/usr/bin/env bash
# DFlash online hybrid training: from-scratch draft, log-EAL + prefix-weighted
# forward KL (1:1), tree/prefix-sharing verifier labels, 2-GPU DDP.
# One epoch per process launch, checkpoint + full acceptance eval every epoch.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env.sh"

RUN_ROOT="${RUN_ROOT:-$ROOT/qwen3_4b_dflash_300k/online_rl/runs/dflash_hybrid_ddp2}"
exec "$SPECULATORS_PYTHON" "$ROOT/qwen3_4b_dflash_300k/online_rl/run_epoch_pipeline.py" \
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
  --lr 3e-4 \
  --muon-lr 3e-3 \
  --train-data-ratio 0.999 \
  --num-workers 12 \
  --checkpoint-freq 0.01 \
  --nproc-per-node 2 \
  --vllm-python "$VLLM_PYTHON" \
  --eval-tag dflash_hybrid \
  --eval-port 8001 \
  --eval-cuda-visible-devices "${EVAL_GPU:-1}" \
  --eval-gpu-memory-utilization 0.5
