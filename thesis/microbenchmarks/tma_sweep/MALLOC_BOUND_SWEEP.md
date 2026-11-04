# Full CPU-bound system-memory sweep

The 2026-11-04 diagnostic run found that per-buffer `mbind(MPOL_BIND)` kept
all tested input pages on the Grace CPU node. This sweep uses that mechanism:
page-aligned `posix_memalign`, explicit range binding before CPU initialization,
and the original system pointer in all three GPU kernels. No CUDA registration
or input copy into HBM is added. The output allocation remains in HBM.

```bash
conda activate fa3
python thesis/scripts/run_malloc_bound_sweep.py --plan
python -u thesis/scripts/run_malloc_bound_sweep.py
```

The full grid is 4 KiB through 16 GiB in powers of two, the original seven
tiles, and only configurations that fit whole tiles. It contains **145
allocation groups / 435 kernel measurements**, with 128 threads per block.

The runner reuses and instruments the existing benchmark and diagnostic helper
in a new output directory. It never overwrites those source files or past runs.
It archives the generated CUDA, original CUDA, Python runners, compiler/GPU
identity and exact commands. Effective allocation policy is required to be
`MPOL_BIND` exclusively to the CPU node. Kernel output correctness and physical
page placement are separate checks.

`cpu_residency_verified=1` requires all input pages on the CPU node at three
snapshots: before the GPU, immediately after the GPU before CPU verification,
and after CPU verification. The last snapshot has its own four CSV fields.
Worker logs include specific NUMA node IDs and negative per-page query statuses.
Snapshots do not prove uninterrupted residency throughout the kernel.

Outputs:

```text
thesis/outputs/malloc_bound_sweep_<UTC timestamp>_<unique suffix>/
  results.csv                 # source=malloc_bound, distinct from plain malloc
  summary.md                  # PASS only for all 435 correct, CPU-verified rows
  manifest.json               # configuration and resume guards
  cases/<bytes>_<tile>/        # raw CSV, log, command, worker return code
  build.log
  environment.txt
  status.json
thesis/outputs/malloc_bound_sweep_<UTC timestamp>_<unique suffix>.tar.gz
thesis/outputs/plots/malloc_bound_sweep_plots_DD_MM_YY/
```

Seven PNG figures are generated automatically from **this run's results.csv**:
four time comparisons (KiB/MiB/GiB sizes and tiles), a speedup curve, a tile
heatmap, and an effective input-rate figure. Athens dates are used for plots;
existing plot directories get `_02`, `_03`, etc. No PDFs are produced. Original
four-allocation plotting still produces its existing 39 figures.

Matplotlib and NumPy are checked before GPU use. To collect data without them:

```bash
python -u thesis/scripts/run_malloc_bound_sweep.py --skip-plots
```

To resume an interrupted run with the same code, configuration, compiler and GPU:

```bash
python -u thesis/scripts/run_malloc_bound_sweep.py --resume thesis/outputs/malloc_bound_sweep_THE_EXACT_RUN
```

Completed case files are validated before reuse. GPU faults, timeouts and
signals stop the run. Results are aggregated after each completed group, and
the finalizer saves the summary and diagnostic archive even on a build failure
or interruption after run creation. The shared sweep lock and GPU process
check prevent overlapping runs from this repository.

To regenerate only the plots later:

```bash
python thesis/scripts/plots/tma_plots_first.py thesis/outputs/malloc_bound_sweep_THE_EXACT_RUN/results.csv --malloc-bound-only
```

Do not overwrite the original plain-malloc measurements or silently splice
this allocation into the old CSV. These are single launches with rotating
kernel order, no warmup and no cache control, matching the original sweep's
timing protocol. Repetitions are needed for final performance conclusions.
