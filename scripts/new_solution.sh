#!/usr/bin/env bash
# Initialize a solution file for a challenge/language from its starter template.
#
# Usage:
#   scripts/new_solution.sh <challenge_dir> <cuda|triton|pytorch>
#
# Example:
#   scripts/new_solution.sh challenges/easy/1_vector_add triton
#   # -> creates challenges/easy/1_vector_add/solution/solution.py (copied from starter.triton.py)
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: scripts/new_solution.sh <challenge_dir> <cuda|triton|pytorch>"
    exit 1
fi

CHALLENGE="$1"; LANGUAGE="$2"

case "$LANGUAGE" in
    cuda)    STARTER="starter.cu";         OUT="solution.cu" ;;
    triton)  STARTER="starter.triton.py";  OUT="solution.py" ;;
    pytorch) STARTER="starter.pytorch.py"; OUT="solution.pytorch.py" ;;
    *) echo "unsupported language: $LANGUAGE (choose cuda|triton|pytorch)"; exit 1 ;;
esac

SRC="$CHALLENGE/starter/$STARTER"
DST_DIR="$CHALLENGE/solution"
DST="$DST_DIR/$OUT"

[[ -f "$SRC" ]] || { echo "template not found: $SRC"; exit 1; }
mkdir -p "$DST_DIR"
if [[ -e "$DST" ]]; then
    echo "$DST already exists; not overwriting."
else
    cp "$SRC" "$DST"
    echo "created $DST — implement the solve() function inside."
    echo "then judge with: scripts/judge.sh $CHALLENGE $LANGUAGE"
fi
