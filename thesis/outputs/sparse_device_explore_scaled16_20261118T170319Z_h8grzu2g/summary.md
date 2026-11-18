# GH200 sparsity exploration at 1/16 scale

- Passed device runs: 256 / 256.
- Failed: 0. Timed out: 0.
- Recorded device worker wall time: 386.5 s.

## Full 1%–100% grid projection at 1/16 scale

Same scaled shapes, backend policy, warmups, repeats and fresh workers. Each full-grid cell uses the wall time of the nearest exploration point for that shape and strategy. Cells nearest a failed, timed-out or pending point are unestimated. This is an approximation, not a bound.

| M×K×N | Estimated cells / 10,000 | Hours for estimated cells |
|---|---:|---:|
| 128×128×128 | 10,000 | 8.21 |
| 256×256×256 | 10,000 | 8.22 |
| 1024×1024×1024 | 10,000 | 8.37 |
| 2048×2048×2048 | 10,000 | 8.65 |
| 256×1024×256 | 10,000 | 8.27 |
| 1024×256×1024 | 10,000 | 8.27 |
| 128×2048×128 | 10,000 | 8.26 |
| 2048×128×2048 | 10,000 | 8.25 |

Estimated coverage: **80,000 / 80,000** cells. Projected wall time for covered cells: **66.49 h**.
