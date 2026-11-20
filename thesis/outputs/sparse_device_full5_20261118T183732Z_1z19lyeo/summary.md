# GH200 full 5% density sweep

- Passed device runs: 6400 / 6400.
- Failed: 0. Timed out: 0.
- Recorded device worker wall time: 94416.5 s.

| M×K×N | Configs passed | Device workers recorded | Worker wall (s) |
|---|---:|---:|---:|
| 2048×2048×2048 | 400/400 | 800/800 | 1212.89 |
| 4096×4096×4096 | 400/400 | 800/800 | 1391.39 |
| 2048×32768×2048 | 400/400 | 800/800 | 2010.16 |
| 4096×16384×4096 | 400/400 | 800/800 | 2204.89 |
| 16384×4096×16384 | 400/400 | 800/800 | 3696.17 |
| 32768×2048×32768 | 400/400 | 800/800 | 5842.90 |
| 16384×16384×16384 | 400/400 | 800/800 | 10196.09 |
| 32768×32768×32768 | 400/400 | 800/800 | 67862.03 |

Use device_times.csv for per-config timings and the plotter for the three heatmaps. Incomplete pairs have no speed ratio.
