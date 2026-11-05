# Warmed and sampled TMA sweep

- Completed allocation groups: 580 / 580
- Expected aggregate measurements: 1740
- Recorded measurements: 1740
- Correct outputs (pre/post sample-loop validation): 1740
- Failed or not run: 0
- Missing measurements: 0
- Expected raw timing samples: 52200
- Recorded raw timing samples: 52200
- CPU placement verified at all four snapshots: 1305 / 1305
- Rows with within-run CV > 10%: 24
- Rows with first sample > 1.5 x median: 1
- Overall correctness/placement: PASS

| Source | Correct / recorded | CPU placement verified / correct host rows |
|---|---:|---:|
| hbm | 435 / 435 | 0 / 0 |
| pinned | 435 / 435 | 435 / 435 |
| registered | 435 / 435 | 435 / 435 |
| malloc_bound | 435 / 435 | 435 / 435 |

kernel_ms is the median; raw samples and p25/p75/p95, min/max, mean, sample SD and CV are retained.
Each sample round runs all three kernels once. All six order permutations occur equally.
Two untimed correctness launches per kernel bracket warmup/sampling; warmup launches are additional.
Placement snapshots: initial, after warmup, after samples before verification, after final verification.
Samples reuse the same input/output buffers after warmup; caches are NOT flushed. This is a warmed-buffer protocol.
Small inputs can be cache-resident. Effective input GB/s is not physical DRAM/NVLink bandwidth.
CUDA events include the complete kernel, including output stores to HBM.
IQR is within-run sample dispersion, not a confidence interval or independent-run replication.
CV and first-sample thresholds are descriptive diagnostics; correctness PASS does not assert statistical timing quality.
