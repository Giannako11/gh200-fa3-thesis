#!/usr/bin/env bash
set -Eeuo pipefail
runner="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/run_tma_targeted_profile.py"
if [[ "${CONDA_DEFAULT_ENV:-}" != fa3 ]]; then
    if [[ ! -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        echo 'Activate the fa3 conda environment before running this script.' >&2
        exit 2
    fi
    set +u
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate fa3
    set -u
fi
exec python -u "$runner" "$@"
