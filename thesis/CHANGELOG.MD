# this is my changelog, here i keep track of changes i make by the end of the day

## 2026-10-29 — TMA / CPU memory microbenchmark

- Added standalone CUDA 12.8/Hopper TMA probes under `thesis/microbenchmarks/tma_cpu/`
  and `thesis/scripts/run_tma_cpu.sh`: HBM control, mapped pinned host memory,
  and CPU-first-touch registered memory, each compared with ordinary GPU loads.
- Records correctness, CUDA failure stage, NUMA residency queries, source/code
  hashes, PTX instruction evidence and optional cache-sensitive timing samples.
- Added CPU-only controller tests. CUDA compilation and GH200 execution remain
  to be validated on grace2; these changes are AI-assisted code for review.
