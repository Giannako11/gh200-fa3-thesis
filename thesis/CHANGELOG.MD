# this is my changelog, here i keep track of changes i make by the end of the day

## 2026-10-15T11:51:50+00:00 — Qwen baseline
- Run: `thesis/outputs/qwen_baseline_20261015_091635_497401787`
- Exit code: 130 (0 = success).

## 2026-10-20T23:58:36+00:00 — Qwen baseline
- Run: `thesis/outputs/qwen_baseline_20261020_234840_957642857`
- Exit code: 0 (0 = success).

## 2026-10-29 — TMA test

- Set up a small CUDA test to check if TMA can read host memory on the GH200.
- Used HBM as a reference, plus two host allocations: `cudaHostAllocMapped`
  and `mmap` with `cudaHostRegisterMapped`.
- Added checks with regular GPU loads, a comparison of every output value,
  and page-location checks before and after the GPU kernels.
- Need to try it on grace2.

## 2026-10-30 — First run on grace2

- Ran the 4 KiB test with CUDA 12.8.1. All six cases passed.
  The generated PTX also contains the tensor TMA instruction.
- The host pages were on CPU NUMA node 0 before and after the GPU kernels.
- The `mbind` request failed with `errno=22`, but the page-location checks
  and the data comparisons passed. --> probably irrelevant for now FLAG this 
  in case theres a problem (could be a bug with passing 0 max nodes)
- Results: `thesis/outputs/tma_cpu_20261030T123257_533342Z/`.
- Next: fix the `maxnode` argument and rerun as said before. 

## 2026-11-02 — TMA benchmark template

- Uploaded the CUDA benchmark, Bash runner and a short README for the full sweep.
- Added a progress bar, CSV results and a run summary.
- This is a general template for now. Waiting for GPU access to run smoke tests
  and refine it before the final run.

## 2026-11-03 — Plain malloc case

- Prepared the CUDA file with plain `malloc` input for manual upload.
- Updated the runner and counts for four allocations: 580 groups, 1,740 measurements.
- The new case uses the original host pointer without CUDA registration or `mbind`.
- Waiting for GPU access to compile and run smoke tests.
