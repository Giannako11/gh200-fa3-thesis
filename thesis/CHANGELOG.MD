# this is my changelog, here i keep track of changes i make by the end of the day

## 2026-10-29 — TMA / CPU memory microbenchmark

- Added standalone CUDA 12.8/Hopper TMA probes under `thesis/microbenchmarks/tma_cpu/`
  and `thesis/scripts/run_tma_cpu.sh`: HBM control, mapped pinned host memory,
  and CPU-first-touch registered memory, each compared with ordinary GPU loads.
- Records correctness, CUDA failure stage, NUMA residency queries, source/code
  hashes, PTX instruction evidence and optional cache-sensitive timing samples.
- Added CPU-only controller tests. CUDA compilation and GH200 execution remain
  to be validated on grace2; these changes are AI-assisted code for review.

## 2026-10-30 — TMA code annotations

- Added Greek stage comments to `thesis/microbenchmarks/tma_cpu/tma_cpu.cu`:
  (1) host allocation and known values, (2) ordinary GPU-load control,
  (3) TMA descriptor/transfer/wait/output, (4) element-by-element verification,
  and (5) NUMA page queries before and after the kernels.
- Removed the introductory AI-related source comment. Executable code, trial
  parameters and result formats are unchanged; GH200 validation is still pending.
