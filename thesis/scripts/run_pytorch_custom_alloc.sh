#!/usr/bin/env bash
set -euo pipefail
runner="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/run_pytorch_custom_alloc.py"
exec python -u "$runner" "$@"
