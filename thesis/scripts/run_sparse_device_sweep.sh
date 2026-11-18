#!/usr/bin/env bash
set -euo pipefail
thesis_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python -u "$thesis_dir/microbenchmarks/sparse_device_sweep/run.py" "$@"
