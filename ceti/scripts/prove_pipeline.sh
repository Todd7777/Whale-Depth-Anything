#!/usr/bin/env bash
# One-command proof: underwater RGB → depth (relative + metric)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${REPO_ROOT}/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON=python3
fi

echo "CETI pipeline proof (use --quick for relative-depth-only)"
exec "$PYTHON" ceti/scripts/prove_pipeline.py "$@"
