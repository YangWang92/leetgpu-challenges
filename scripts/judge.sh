#!/usr/bin/env bash
# Launcher for the local judge: activates the venv, puts nvcc on PATH, then runs local_runner.py.
#
# Usage:
#   scripts/judge.sh <challenge_dir> <cuda|triton|pytorch> [action] [extra args...]
#
# Examples:
#   scripts/judge.sh challenges/easy/1_vector_add cuda
#   scripts/judge.sh challenges/easy/1_vector_add triton perf
#   scripts/judge.sh challenges/easy/1_vector_add pytorch all
#
# action: test (default) | perf | all
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"

if [[ $# -lt 2 ]]; then
    echo "usage: scripts/judge.sh <challenge_dir> <cuda|triton|pytorch> [test|perf|all]"
    exit 1
fi

CHALLENGE="$1"; LANGUAGE="$2"; ACTION="${3:-test}"; shift $(( $# < 3 ? $# : 3 ))

# Activate the venv.
if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate"
else
    echo "warning: $REPO_ROOT/.venv not found; set up the environment first (see scripts/README_LOCAL.md)"
fi

# Put nvcc on PATH (needed for the cuda language).
export PATH="$CUDA_HOME/bin:$PATH"

exec python "$REPO_ROOT/scripts/local_runner.py" "$CHALLENGE" \
    --language "$LANGUAGE" --action "$ACTION" "$@"
