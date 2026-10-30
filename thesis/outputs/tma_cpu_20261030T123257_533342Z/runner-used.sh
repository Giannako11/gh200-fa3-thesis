#!/usr/bin/env bash
# Small standalone experiment; no installs, Qwen downloads, or FA3 rebuilds.
set -Eeuo pipefail
repo_root="$(git -C "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)"
set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate fa3
set -u
selected_cuda=''
for candidate in "$CONDA_PREFIX/cuda-12.8.1" /usr/local/cuda-12.8; do
    if [[ -x "$candidate/bin/nvcc" ]] && [[ "$("$candidate/bin/nvcc" --version)" == *'release 12.8,'* ]]; then
        selected_cuda="$candidate"
        break
    fi
done
[[ -n "$selected_cuda" ]] || { echo 'CUDA Toolkit 12.8 not found; no installation attempted.' >&2; exit 2; }
exec python "$repo_root/thesis/microbenchmarks/tma_cpu/run.py" --toolkit "$selected_cuda" "$@"
