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
