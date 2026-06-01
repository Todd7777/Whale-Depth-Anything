#!/usr/bin/env bash
# M5 Max: skip setup — only ensure data + train (after setup already passed)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

exec bash "$(dirname "$0")/mac_train_bulletproof.sh"
