# GH200 block-selected decode attention placement

- Status: PASS. Both placements matched each other and the FP32 reference.
- K/V physical HBM delta: 32 MiB in HBM, 0 MiB in Grace.
- Grace node 0 before and after: 256 pages per K/V tensor.
- Timing uses CUDA events around the two decode kernels, warmed buffers, alternating order, and no explicit cache eviction.

| Blocks read | HBM median (µs) | Grace median (µs) | Grace / HBM |
|---:|---:|---:|---:|
| 8/64 | 17.968 | 50.672 | 2.820× |
| 16/64 | 19.184 | 63.936 | 3.333× |
| 32/64 | 21.424 | 91.472 | 4.270× |
| 64/64 | 27.808 | 145.664 | 5.238× |
