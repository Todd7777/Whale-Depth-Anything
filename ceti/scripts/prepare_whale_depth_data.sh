#!/usr/bin/env bash
# Build whale/underwater depth training lists from REAL field imagery
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

echo "Prefer real imagery curation (DAVIS + HF datasets):"
exec bash ceti/scripts/curate_underwater_field.sh "$@"
