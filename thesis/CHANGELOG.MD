# this is my changelog, here i keep track of changes i make by the end of the day

## 2026-11-16 — Mixed K/V placement completed

- The smoke and full runs passed on GH200. The full run recorded 280/280
  samples across seven layouts. Outputs matched the all-HBM baseline and
  the FP32 reference. Grace page placement and pool release checks passed.
- With 32 MiB of HBM saved in each layout, keeping all active blocks in HBM
  took 28.24 µs, splitting active blocks evenly took 94.34 µs, and keeping
  all active blocks in Grace took 138.86 µs. The block list was fixed.
- Results: [full mixed-placement run](outputs/sparse_attention_mixed_20261115T220837Z_va6lhsqu/summary.md).

## 2026-11-15 — Sparse attention K/V placement

- The all-HBM vs. all-Grace benchmark passed on GH200. Moving the K/V cache
  saved 32 MiB of HBM. Grace-backed decode latency was 2.82–5.24× higher,
  depending on the number of blocks read. All 240 full-run samples are
  [recorded here](outputs/sparse_attention_placement_20261115T185926Z_36wtnrid/summary.md).
- Added mixed K/V placement with 64 of 128 blocks read in every layout.

## 2026-11-15 — Step 3: selective CUDA tensor allocation completed

- Added Grace DRAM allocation and free callbacks, a scoped `MemPool` wrapper
  for stock PyTorch 2.11, and a GH200 validation runner.
- GH200 validation passed. In the same process, the default 32 MiB input and
  output each used 32 MiB of HBM. The 32 MiB Grace input added zero HBM use
  and all 512 pages stayed on CPU NUMA node 0 before and after GPU work.
- Adding the two inputs on the GPU produced 12 in every output element.
  Pool teardown called `grace_free` for the allocated pointer and live bytes
  returned to zero.
- Results: [GH200 validation](outputs/torch_custom_alloc_20261115T182058Z_qduryofk/result.json).
- Follow-up: test placement of selected sparse attention buffers in Grace DRAM.

## 2026-11-07 — Step 2: GH200 runtime validation completed

- Built stock and patched PyTorch from source and passed all 16 GH200 runtime
  checks. The 32 MiB host-backed allocation had all 512 pages on CPU NUMA
  node 0, with no increase in HBM use. GPU arithmetic returned correct results.
- Results: [GH200 validation summary](outputs/torch_host_malloc_20261107T171406Z_o7jc7yft/summary.md).

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
