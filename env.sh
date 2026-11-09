#!/usr/bin/env bash
# Portable environment for this release.
# Export the two interpreters before sourcing (or edit the defaults below):
#   export SPECULATORS_PYTHON=/path/to/speculators-env/bin/python
#   export VLLM_PYTHON=/path/to/vllm-env/bin/python
# then: source ./env.sh

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export SPECULATORS_ROOT="$ROOT"
export SPECULATORS_SRC="${SPECULATORS_SRC:-$ROOT/speculators-src}"
export SPECULATORS_PYTHON="${SPECULATORS_PYTHON:-python3}"
export VLLM_PYTHON="${VLLM_PYTHON:-python3}"

if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${SPECULATORS_SRC}/src:${PYTHONPATH}"
else
    export PYTHONPATH="${SPECULATORS_SRC}/src"
fi

# Local vLLM endpoints should bypass any system proxy settings.
unset ALL_PROXY all_proxy HTTP_PROXY HTTPS_PROXY http_proxy https_proxy || true

echo "SPECULATORS_SRC=${SPECULATORS_SRC}"
echo "SPECULATORS_PYTHON=${SPECULATORS_PYTHON}"
echo "VLLM_PYTHON=${VLLM_PYTHON}"
