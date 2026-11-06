# this is my changelog, here i keep track of changes i make by the end of the day

## 2026-11-06 — Source-build disk thresholds

- Kept the native PyTorch source-patch/build method. Lowered the initial free
  disk guard from 80 to 30 GiB, as requested; this is not a measured build size.
- Added a 0.5-second watchdog with an 8 GiB free-space stop threshold during
  build commands. It terminates only its own command group and retains logs
  and partial artifacts. No automatic deletion or package cleanup.

## 2026-11-06 — Step 2: opt-in malloc in PyTorch

- Prepared a source patch pinned to PyTorch v2.11.0 commit
  `70d99e998b4955e0049d13a98d77ae1b14db1f45`, redirecting native backing blocks
  and uncached allocations to ordinary malloc with verified node-0 placement.
- Added process-frozen OFF-by-default activation, matching free bookkeeping,
  conservative GPU synchronization, range-policy cleanup and an allocation cap.
- Added isolated stock/patched wheel builds, fresh conda prefixes, provenance
  manifests, a 16-case GH200 runner, telemetry, progress and retained diagnostics.
- Added CPU-only helper tests with explicit CUDA/NUMA test doubles and patch
  consistency checks. These do not establish GH200 runtime compatibility.
- Actual PyTorch/CUDA compilation and the full GPU run must still execute on
  Grace2. Step 2 remains in progress; the existing fa3 installation is not edited.

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

## 2026-11-03 — First TMA plots

- Added `thesis/scripts/plots/tma_plots_first.py` for the 34 agreed figures.
- Saves separate and combined plots in PNG/PDF, plus an internal checks CSV.
- Checked the plots with synthetic data, including missing and failed cases.
- Ready to use with the full sweep CSV when GPU measurements are available.

## 2026-11-03 — Effective input rate plots

- Added five input-rate figures in GB/s, bringing the total to 39.
- Uses input bytes divided by the full kernel time, including HBM output stores.
- Kept the placement markers and gaps for failed or missing measurements.

## 2026-11-03 — Plot output folders

- Switched the plotting script to PNG only.
- Saves the 39 PNGs in `thesis/outputs/plots/tma_plots_first_plots_DD_MM_YY/`.
- Repeated runs get a numbered suffix. Internal notes and checks stay in `_checks/`.
