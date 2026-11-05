# Universal warmed and sampled GH200 sweep

```bash
bash thesis/scripts/run_tma_sampled_sweep.sh
```

The default grid uses all 23 powers-of-two input sizes from 4 KiB to 16 GiB,
all seven original tiles that fit whole tiles, and four allocations:

| CSV source | Input allocation |
|---|---|
| `hbm` | `cudaMalloc` |
| `pinned` | mapped `cudaHostAlloc` |
| `registered` | CPU-bound `mmap`, then mapped `cudaHostRegister` |
| `malloc_bound` | page-aligned `posix_memalign`, then explicit per-buffer `mbind(MPOL_BIND)` |

There are **580 allocation groups, 1,740 median rows, and 52,200 raw timing
samples** with defaults. The corrected CPU-bound system allocation replaces
plain `malloc` in this new protocol. Original runs and plotting modes are kept.
No old measurements are combined with this run.
The Bash entry point activates the existing `fa3` environment and delegates to
the Python runner. It does not install packages or change machine settings.

## Measurement protocol

Each group uses a fresh worker process and one input/output allocation pair.
All three unchanged GPU kernels are measured on that pair, with 128 threads.
CUDA events are created and their record/synchronize path is primed before
the warmup and sample loops.

1. Query initial page placement and allocation policy. For `malloc_bound`,
   require per-buffer `MPOL_BIND` exclusively to the CPU node.
2. Launch each kernel once without timing, clearing the output beforehand and
   verifying every output element. These three launches also force first use
   of every kernel outside the measured region.
3. Perform **10 untimed warmup launches per kernel**, then synchronize.
4. Query placement after warmup, before samples.
5. Take **30 separately timed CUDA-event samples per kernel**. Every round
   runs all three kernels once. All six permutations occur five times, so
   every kernel occupies every position equally often. No output resets,
   correctness copies, page queries, or allocations occur between timed
   samples. Each timed interval covers one complete kernel, including output
   stores to HBM; event synchronization is outside the timed interval.
6. Query placement after the samples, before final correctness validation.
7. Clear the output and validate each kernel again, untimed, then query final
   page placement.

Output checks bracket the sample loop; output is not copied to the CPU after
every timed launch. `cpu_residency_verified=1` requires all four host placement
snapshots to be on the CPU node with no unknown pages or query errors.
Snapshots are not continuous tracing.

This is a **warmed-buffer, no-cache-flush** protocol. The input/output buffers
are reused. Small buffers can be served from caches, so effective input GB/s
must not be interpreted as physical Grace DRAM or NVLink bandwidth. Samples
from one allocation/process are within-run repetitions, not independent
process/run replications. GPU clocks are not locked. Raw samples and variation
diagnostics are retained so first-sample effects and variability remain visible.

## Files and plots

```text
thesis/outputs/tma_sampled_sweep_<UTC timestamp>_<suffix>/
  results.csv          # 1740 rows; kernel_ms = median
  samples.csv          # 52200 raw sample times and positions
  summary.md
  manifest.json        # exact protocol, source hashes, compiler/GPU identity
  environment.txt
  sampled.cu           # exact CUDA source compiled for this run
  build.log
  cases/<bytes>_<tile>_<source>/
    rows.csv
    samples.csv
    worker.log
    command.json
    done.json          # hashes of both case CSVs
  plots.log
  status.json
thesis/outputs/tma_sampled_sweep_<UTC timestamp>_<suffix>.tar.gz
thesis/outputs/plots/tma_sampled_sweep_plots_DD_MM_YY/
```

`results.csv` includes min, p25, median, p75, p95, max, mean, sample standard
deviation, CV, first sample, actual sample count, warmup count, and four sets of
page-location counters. No samples are discarded. The original full grids and
CUDA kernels are reused; the allocation changes were validated separately on
GH200. The generated CUDA and all helper sources are archived.

The **39 PNGs** are generated automatically from this run's aggregate CSV.
Times, ratios and rates use medians; time bars have p25-to-p75 whiskers. IQR
describes sample dispersion, not a confidence interval. No PDFs are produced.
Plots use Athens dates and preserve prior outputs with numbered suffixes.
The diagnostic archive contains data/logs/sources, while PNGs live separately
in `thesis/outputs/plots/`.

Overall PASS requires complete correct rows and verified host placement.
The summary separately reports CV >10% and first-sample >1.5x median counts as
descriptive timing diagnostics. PASS is not a statistical stability guarantee.

## Commands

Plan only (no GPU, plotting libraries, or output directory required):

```bash
python thesis/scripts/run_tma_sampled_sweep.py --plan
```

Optional smoke: 1 MiB, 32x32, all four allocations:

```bash
python -u thesis/scripts/run_tma_sampled_sweep.py --smoke --skip-plots
```

Explicit settings (sample count must be at least six and divisible by six):

```bash
python -u thesis/scripts/run_tma_sampled_sweep.py --warmup 10 --samples 60
```

Collect without plotting dependencies:

```bash
python -u thesis/scripts/run_tma_sampled_sweep.py --skip-plots
```

Resume with the same settings, sources, compiler and GPU:

```bash
python -u thesis/scripts/run_tma_sampled_sweep.py --resume thesis/outputs/tma_sampled_sweep_EXACT_RUN
```

If the original run used custom `--warmup`, `--samples`, or `--smoke`, repeat
those options on resume. Completed CSVs are hash-checked and validated against
raw samples. The raw aggregate is rebuilt on resume/finalization to prevent
duplicate rows after interrupted append operations. The shared sweep lock and
GPU-process check prevent overlapping work; GPU faults/timeouts/signals stop
the run. An archive is finalized even on build failure or interruption after
the guarded run has started. At the end, send the `SEND_THIS_FILE` archive.

Regenerate the plots later:

```bash
python thesis/scripts/plots/tma_plots_first.py thesis/outputs/tma_sampled_sweep_EXACT_RUN/results.csv --sampled
```
