# Full malloc_bound sweep

- Expected measurements: 435
- Recorded measurements: 435
- Correct outputs: 435
- Failed or not run: 0
- Missing measurements: 0
- CPU residency verified at all three checks: 435 / 435
- Overall: PASS

Mechanism: page-aligned posix_memalign + explicit per-buffer mbind(MPOL_BIND) to the CPU node.
Effective allocation policy is queried and required to bind exclusively to that node.
Source label: malloc_bound. No cudaHostRegister; the GPU uses the system pointer.
Placement checks: before GPU, after GPU before CPU verification, and after verification.
Checks are snapshots, not continuous residency tracing.
Three GPU kernels per configuration; 128 threads; one launch each; no warmup or cache control.
Time includes HBM output stores. Use repeated measurements for final performance claims.
Do not replace plain malloc rows in the original sweep with these measurements.
