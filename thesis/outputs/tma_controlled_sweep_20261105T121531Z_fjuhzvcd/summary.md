# Controlled TMA sweep

- Validated worker processes: 3480 / 3480
- Validated timing samples: 313200 / 313200
- Complete pooled rows: 3480 / 3480
- Overall output/placement/conditioning: PASS

gpu_evicted: eviction -> timed target.
self_primed: eviction -> one untimed SAME target -> timed target.
Each kernel is sampled in an isolated block. Block order cycles across independent allocations.
Events include the full copy kernel and its HBM output writes. Conditioning/primer are excluded.
GPU eviction is empirical conditioning, not guaranteed hardware invalidation. Grace CPU caches are not flushed.
Page placement is checked at four snapshots per kernel, not continuously.
Medians/IQR pool equal samples from independent allocations. IQR is not a confidence interval.
Effective input GB/s is not physical DRAM or NVLink bandwidth. No outliers are deleted.
