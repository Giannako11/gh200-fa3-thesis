# TMA sweep summary

- Expected measurements: 1740
- Recorded measurements: 1740
- Correct outputs: 1740
- Failed or not run: 0
- Missing measurements: 0
- Correct host outputs with CPU page placement verified: 880 / 1305

## Results by source and kernel

- hbm/tma_shared: 145 / 145 correct
- malloc/loads_shared: 145 / 145 correct
- malloc/loads_direct: 145 / 145 correct
- malloc/tma_shared: 145 / 145 correct
- pinned/loads_shared: 145 / 145 correct
- pinned/loads_direct: 145 / 145 correct
- registered/loads_shared: 145 / 145 correct
- hbm/loads_shared: 145 / 145 correct
- registered/loads_direct: 145 / 145 correct
- hbm/loads_direct: 145 / 145 correct
- registered/tma_shared: 145 / 145 correct
- pinned/tma_shared: 145 / 145 correct

One timed launch per row. No warmup or cache control.
GPU events include the complete copy kernel and its output writes.
Correct output and CPU page placement are separate checks.
