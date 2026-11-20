# GH200 sparsity exploration at 1/16 scale

- Passed device runs: 8 / 256.
- Failed: 0. Timed out: 0.
- Recorded device worker wall time: 11.9 s.

## Full 1%–100% grid projection at 1/16 scale

Same scaled shapes, backend policy, warmups, repeats and fresh workers. Each full-grid cell uses the wall time of the nearest exploration point for that shape and strategy. Cells nearest a failed, timed-out or pending point are unestimated. This is an approximation, not a bound.

| M×K×N | Estimated cells / 10,000 | Hours for estimated cells |
|---|---:|---:|
| 128×128×128 | 0 | 0.00 |
| 256×256×256 | 0 | 0.00 |
| 1024×1024×1024 | 625 | 0.51 |
| 2048×2048×2048 | 625 | 0.53 |
| 256×1024×256 | 0 | 0.00 |
| 1024×256×1024 | 625 | 0.50 |
| 128×2048×128 | 0 | 0.00 |
| 2048×128×2048 | 625 | 0.52 |

Estimated coverage: **2,500 / 80,000** cells. Projected wall time for covered cells: **2.06 h**.
A full-run estimate is unavailable for the uncovered cells.
