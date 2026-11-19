#!/usr/bin/env bash
# 300 real-text prompts x 7 context lengths; existing fa3 environment/model.
set -Eeuo pipefail
repo_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$repo_root"
if [[ "${CONDA_DEFAULT_ENV:-}" != fa3 ]]; then
    if [[ ! -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        echo 'Activate the existing fa3 environment first.' >&2
        exit 2
    fi
    set +u
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate fa3
    set -u
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export THESIS_QWEN_REPO="$repo_root"
engine="$repo_root/thesis/qwen_sparsity/run.py"
if [[ "${1:-}" == --resume && -f "${2:-}/runner_source/run.py" ]]; then
    engine="$(realpath "$2")/runner_source/run.py"
fi
exec python -u "$engine" "$@"
