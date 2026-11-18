# Grace CPU vs. Hopper GPU: 128-case matrix multiply exploration

This pilot estimates the cost of a complete density-by-density sweep. It
uses one deterministic row-uniform sparsity pattern, FP32 and exactly 128
configs: densities 1%, 10%, 50% and 100% for each input at eight shapes.
Each config runs on both devices. The eight 100% × 100% cases run first,
starting at 32K × 32K × 32K.

| A shape | B shape | C shape |
|---|---|---|
| 2K×2K, 4K×4K, 16K×16K, 32K×32K | Same dimensions as A | Same dimensions as A |
| 4K×16K | 16K×4K | 4K×4K |
| 16K×4K | 4K×16K | 16K×16K |
| 2K×32K | 32K×2K | 2K×2K |
| 32K×2K | 2K×32K | 32K×32K |

K = 1024. Matrices use implicit or explicit ones at the selected nonzero
positions. Both workers regenerate the same inputs from fixed seeds and
compare up to 64 output entries to an independent FP32 reference. No
matrices are written to disk.

## Run on Grace2

From the repository root on an idle Grace2 with CUDA PyTorch, NumPy and SciPy:

```bash
conda activate fa3
bash thesis/scripts/run_sparse_device_sweep.sh --plan
bash thesis/scripts/run_sparse_device_sweep.sh --smoke
bash thesis/scripts/run_sparse_device_sweep.sh
```

Smoke exercises all four representation strategies with small test
matrices. The exploration defaults to 16 CPU/BLAS threads, one warmup and
one timed iteration per device, and 1,200 seconds per device worker.
Use `--threads`, `--warmups`, `--repeats` and `--timeout` to change those
settings. SciPy sparse routines and NumPy BLAS may use different numbers
of threads. Use the same environment and options for comparable timings.
A timeout preserves the latest stage and lets the other backend and
remaining configs run. Resume with the original options and
`--resume /path/to/run`. Add `--retry-failed` to retry unsuccessful workers.
You can change the timeout on a retry.

## What is timed

- Grace uses NumPy BLAS for dense paths and SciPy for sparse paths.
  Hopper uses PyTorch CUDA. The worker uses the same representation
  policy on both devices.
- A lightweight rule chooses CSR for an operand with density at most 10%.
  CSR×CSR is used when both qualify and the estimated number of intermediate
  products is at most 500 million. Above that limit the less dense operand
  is used as CSR and the other as a strided matrix. The remaining paths use
  dense GEMM, including the 100% × 100% anchors. This rule controls the cost
  of the experiment. Its thresholds are not a CPU/GPU placement policy.
- Static A preparation, process startup, generation, warmups and validation
  are included in the worker wall time used for the sweep projection.
- Per-inference timing begins with an existing dense representation of B
  in HBM and ends with dense C ready in HBM. Grace uses blocking copies
  through pageable host tensors. B format conversion and sparse output
  materialization are timed separately. GPU stages synchronize for clear
  wall-time attribution and also record CUDA event time for compute.
  These are serialized measurements for the selected backends, not an
  optimized inference-engine comparison.

## Outputs

Each run writes `thesis/outputs/sparse_device_explore_*` with:

- `config_times.csv`: total worker time per config, including both devices.
- `device_times.csv`: wall time and timing medians per device and config,
  with timeout/failure status and the last stage reached.
- `steps.csv` and per-worker `*.steps.jsonl`: stage timings for setup,
  warmups, timed iterations, copies, conversion, compute and validation.
  Per-worker files are flushed as the stages finish.
- `estimate.csv` and `summary.md`: estimated hours per shape for the
  80,000-cell sweep with the same backend policy and timing protocol.
  Each cell is estimated from the nearest pilot point of the same shape
  and strategy. Timeouts and failed anchors leave explicit coverage gaps.
  The estimate assumes serial processing and fresh workers and should be
  recalibrated if backend implementations or the benchmark protocol change.
- `manifest.json` and worker logs for reproducibility and diagnosis.
