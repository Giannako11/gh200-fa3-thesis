# Grace CPU vs. Hopper GPU: large-shape density endpoints

This exploration tests the two density endpoints of eight large FP32 matrix
multiplications. The **100% density** case uses two all-one matrices and dense
GEMM. The **0% density (100% sparsity)** case uses two all-zero matrices and the
CSR×CSR path. This second case measures the cost of the zero-input path,
including result materialization. It cannot predict mid-range sparsity timings.

| A shape | B shape | C shape |
|---|---|---|
| 2K×2K, 4K×4K, 16K×16K, 32K×32K | Same dimensions as A | Same dimensions as A |
| 4K×16K | 16K×4K | 4K×4K |
| 16K×4K | 4K×16K | 16K×16K |
| 2K×32K | 32K×2K | 2K×2K |
| 32K×2K | 2K×32K | 32K×32K |

K = 1,024. There are 16 configs and 32 device workers. The dense configs run
first, starting at 32K×32K. Each worker compares up to 64 result entries
against a reference. No matrices are written to disk.

## Run on Grace2

From the repository root on an idle Grace2 with CUDA PyTorch, NumPy and SciPy:

```bash
conda activate fa3
bash thesis/scripts/run_sparse_device_sweep.sh --plan
bash thesis/scripts/run_sparse_device_sweep.sh --smoke
bash thesis/scripts/run_sparse_device_sweep.sh
```

The smoke uses both endpoints at a small shape. Defaults are 16 CPU/BLAS
threads, zero warmups, one timed iteration per device and a 1,200-second
timeout per worker. Set `--threads`, `--warmups`, `--repeats` or `--timeout`
to change those defaults. A timeout leaves the recorded stage logs and
lets the runner move to the next worker. Resume this grid with the original
options and `--resume /path/to/run`. Add `--retry-failed` to retry unsuccessful
workers. Previous exploration folders are separate runs.

## Timings and outputs

Grace uses NumPy BLAS and SciPy. Hopper uses PyTorch CUDA. Both use the
same representation for each endpoint. Wall time per worker includes startup,
matrix creation, preparation, measurement and validation. The end-to-end
iteration starts with B in HBM and ends with dense C in HBM, including actual
copies in the Grace path. Stage logs separate compute, format conversion,
output materialization and copies.

Results are stored under `thesis/outputs/sparse_device_explore_extremes_*`:

- `summary.md` pairs the Grace and Hopper wall times for each endpoint.
- `config_times.csv` records total time per config.
- `device_times.csv` records per-device times, status and the last stage reached.
- `steps.csv` and the per-worker JSONL files record stage times.
- `manifest.json` and worker logs provide settings and diagnostics.

## Full grid at 5% intervals

The same eight shapes are run at the original scale. For each, A and B
independently take densities 5%, 10%, …, 100%. This is a 20×20 grid and
400 configs per shape, for **3,200 configs and 6,400 device workers**.
Runs are ordered from smaller to larger multiplication workloads.

```bash
bash thesis/scripts/run_sparse_device_sweep.sh --full --plan | tail -1
bash thesis/scripts/run_sparse_device_sweep.sh --full --smoke &&
bash thesis/scripts/run_sparse_device_sweep.sh --full
```

The full smoke tests nonzero CSR×CSR, CSR×dense, dense×CSR, and dense GEMM
at small dimensions. Full results are written under
`thesis/outputs/sparse_device_full5_*`. The summary and CSVs are rebuilt after
each completed shape and on a graceful interruption. The per-worker files
are saved throughout the run. Resume with the same measurement options:

```bash
bash thesis/scripts/run_sparse_device_sweep.sh --full --resume /path/to/run
```

Use `--retry-failed` on a resume if you want to retry failed or timed-out
workers. Completed workers are reused. The default timeout is 1,200 seconds
per device worker, and a timed-out worker is reported as such.

The grid uses the pilot's representation policy on both devices. CSR is
used for operands of at most 10% density. The CSR×CSR path also requires
at most 500 million predicted intermediate products. Above that limit,
the less dense operand uses CSR and the other uses dense storage. When
both operands exceed 10% density, the backend uses dense GEMM.
The CSV records the strategy for each config.

### Heatmaps

The run writes three PNGs when Matplotlib is available, each with
eight panels and the measured A and B densities on separate axes:

- `heatmap_compute.png`: log₂ of Hopper CUDA event time divided by Grace
  compute time. Positive values favor Grace.
- `heatmap_transfers.png`: median Grace round-trip copy time in ms. The
  runner sums both directions per timed iteration before taking the median.
- `heatmap_end_to_end.png`: log₂ of the ratio of total Hopper to Grace time.
  Positive values favor Grace. The measurement starts with B in HBM and
  ends with dense C available in HBM.

Missing or failed measurements are gray. The plots can be generated from the
saved CSVs in any Python environment with Matplotlib, without CUDA:

```bash
bash thesis/scripts/run_sparse_device_sweep.sh --plots-only /path/to/run
```
