# GH200 mixed K/V placement

- Status: PASS. All layouts matched the FP32 reference and NUMA placement checks.
- Fixed workload: 64 of 128 blocks read, 64 MiB K/V cache, one query per head.
- GPU events cover the two decode kernels. Warmed buffers, no explicit eviction.
- HBM savings refer to the experimental K/V allocation. The HBM reference source and scratch are outside the allocation deltas.

| Policy | Grace resident | Grace read | HBM saved (MiB) | Median (µs) |
|---|---:|---:|---:|---:|
| all_hbm | 0/128 | 0/64 | 0 | 33.696 |
| balanced_25 | 32/128 | 16/64 | 16 | 64.800 |
| balanced_50 | 64/128 | 32/64 | 32 | 94.336 |
| balanced_75 | 96/128 | 48/64 | 48 | 120.416 |
| all_grace | 128/128 | 64/64 | 64 | 146.320 |
| active_hbm | 64/128 | 0/64 | 32 | 28.240 |
| active_grace | 64/128 | 64/64 | 32 | 138.864 |
