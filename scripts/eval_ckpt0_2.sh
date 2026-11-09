#!/usr/bin/env bash
# Full acceptance evaluation for DFlash ckpt0-2 (hybrid run + offline baseline).
# Launches a temporary vLLM 0.28 server per checkpoint and reads its Prometheus
# counters. Results land in qwen3_4b_dflash_300k/test_data/results/.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env.sh"

RESULTS="$ROOT/qwen3_4b_dflash_300k/test_data/results"
HYBRID_RUN="${HYBRID_RUN:-$ROOT/qwen3_4b_dflash_300k/online_rl/runs/dflash_hybrid_ddp2}"
OFFLINE_ROOT="${OFFLINE_ROOT:-$ROOT/qwen3_4b_dflash_300k/checkpoints}"
EVAL_GPU="${EVAL_GPU:-0}"

run_eval() {
  local ckpt="$1"
  local out="$2"
  if [[ -f "$out/results.json" ]]; then
    echo "skip existing $out"
    return 0
  fi
  echo "evaluating $ckpt -> $out"
  "$SPECULATORS_PYTHON" "$ROOT/qwen3_4b_dflash_300k/test_data/test_dflash_acceptance.py" \
    --draft-model "$ckpt" \
    --vllm-python "$VLLM_PYTHON" \
    --max-samples 0 \
    --seed 0 \
    --temperature 0.0 \
    --concurrency 1 \
    --max-tokens 1024 \
    --max-model-len 8192 \
    --datasets gsm8k humaneval mt-bench \
    --port "${PORT:-8001}" \
    --cuda-visible-devices "$EVAL_GPU" \
    --output-dir "$out"
}

for k in 0 1 2; do
  run_eval "$HYBRID_RUN/checkpoints/$k" "$RESULTS/dflash_hybrid_epoch$k"
  run_eval "$OFFLINE_ROOT/$k" "$RESULTS/dflash_offline_ckpt$k"
done
