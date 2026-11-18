# GH200 sparsity exploration

- Passed device runs: 3 / 256.
- Failed: 0. Timed out: 0.
- Recorded device worker wall time: 113.2 s.

## Full 1%–100% grid projection

Same shapes, backend policy, warmups, repeats and fresh workers. Each full-grid cell uses the wall time of the nearest exploration point for that shape and strategy. Cells nearest a failed, timed-out or pending point are unestimated. This is an approximation, not a bound.

| M×K×N | Estimated cells / 10,000 | Hours for estimated cells |
|---|---:|---:|
| 2048×2048×2048 | 0 | 0.00 |
| 4096×4096×4096 | 0 | 0.00 |
| 16384×16384×16384 | 0 | 0.00 |
| 32768×32768×32768 | 625 | 19.31 |
| 4096×16384×4096 | 0 | 0.00 |
| 16384×4096×16384 | 0 | 0.00 |
| 2048×32768×2048 | 0 | 0.00 |
| 32768×2048×32768 | 0 | 0.00 |

Estimated coverage: **625 / 80,000** cells. Projected wall time for covered cells: **19.31 h**.
A full-run estimate is unavailable for the uncovered cells.
