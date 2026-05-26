#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${CONTEXTFORGE_REPO:-$ROOT_DIR/../contextforge}:$ROOT_DIR/sidecar:${PYTHONPATH:-}"
export CONTEXTFORGE_DB_PATH="${CONTEXTFORGE_DB_PATH:-$ROOT_DIR/contextforge.db}"
export CONTEXTFORGE_MAX_CONTEXT_TOKENS="${CONTEXTFORGE_MAX_CONTEXT_TOKENS:-4096}"

python3 -m contextforge_sidecar

