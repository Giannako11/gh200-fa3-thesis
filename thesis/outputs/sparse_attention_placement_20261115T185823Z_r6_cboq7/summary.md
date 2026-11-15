# GH200 block-selected decode attention placement

- Status: PASS. Both placements matched each other and the FP32 reference.
- K/V physical HBM delta: 32 MiB in HBM, 0 MiB in Grace.
- Grace node 0 before and after: 256 pages per K/V tensor.
- Timing uses CUDA events around the two decode kernels, warmed buffers, alternating order, and no explicit cache eviction.

| Blocks read | HBM median (µs) | Grace median (µs) | Grace / HBM |
|---:|---:|---:|---:|
| 16/64 | 19.168 | 62.208 | 3.245× |
