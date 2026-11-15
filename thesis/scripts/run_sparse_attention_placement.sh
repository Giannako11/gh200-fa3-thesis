#!/usr/bin/env bash
set -euo pipefail
runner="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/run_sparse_attention_placement.py"
exec python -u "$runner" "$@"
