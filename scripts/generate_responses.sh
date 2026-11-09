#!/usr/bin/env bash
# Regenerate the 300k responses with the target model (temperature 0).
# Requires a running vLLM chat endpoint (see start_vllm.sh).
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env.sh"

MODEL="${MODEL:-$ROOT/Qwen3-4B-Instruct}"
exec "$SPECULATORS_PYTHON" "$ROOT/data_pipeline/regenerate_local_responses.py" \
  --data "$ROOT/open_perfectblend_1trun_v2_300k/data/*.parquet" \
  --model "$MODEL" \
  --endpoint "http://127.0.0.1:${PORT:-8000}/v1/chat/completions" \
  --outfile "$ROOT/qwen3_4b_dflash_300k/on_policy.jsonl" \
  --concurrency "${CONCURRENCY:-64}" \
  --max-tokens 8192 \
  --sampling-params '{"temperature":0}' \
  --max-retries 3 \
  --resume
