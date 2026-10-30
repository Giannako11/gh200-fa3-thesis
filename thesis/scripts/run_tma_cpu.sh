#!/usr/bin/env bash
# Ξεκινά τη δοκιμή TMA με το υπάρχον περιβάλλον fa3 και CUDA Toolkit 12.8.
# Ο GPU kernel και η πηγή του παραδείγματος είναι στο thesis/microbenchmarks/tma_cpu/tma_cpu.cu.
set -Eeuo pipefail
# Βρίσκουμε το repo από τη θέση αυτού του αρχείου.
repo_root="$(git -C "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)"
# Χαλαρώνουμε προσωρινά τον έλεγχο μη ορισμένων μεταβλητών για την ενεργοποίηση του Conda.
set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate fa3
set -u
# Διαλέγουμε το Toolkit 12.8 που έχουμε ήδη εγκαταστήσει.
selected_cuda=''
for candidate in "$CONDA_PREFIX/cuda-12.8.1" /usr/local/cuda-12.8; do
    if [[ -x "$candidate/bin/nvcc" ]] && [[ "$("$candidate/bin/nvcc" --version)" == *'release 12.8,'* ]]; then
        selected_cuda="$candidate"
        break
    fi
done
[[ -n "$selected_cuda" ]] || { echo 'CUDA Toolkit 12.8 not found; no installation attempted.' >&2; exit 2; }
# Το Python script αναλαμβάνει compilation, δοκιμές και αποθήκευση των αποτελεσμάτων.
exec python "$repo_root/thesis/microbenchmarks/tma_cpu/run.py" --toolkit "$selected_cuda" "$@"
