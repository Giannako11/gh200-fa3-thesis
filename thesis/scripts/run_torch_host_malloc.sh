#!/usr/bin/env bash
set -Eeuo pipefail
runner="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/run_torch_host_malloc.py"
exec python -u "$runner" "$@"
