# PyTorch host malloc checks

- Result: PASS
- Completed checks: 16 / 16
- Scope: single-GPU eager inference, native allocator, bounded host allocations.
- CUDA tensor storage and logical allocator counters do not imply physical HBM placement.
- Host-to-host CUDA tensor copies use a synchronized CPU copy; GPU arithmetic uses CUDA kernels.
- Timings are diagnostic test durations, not performance benchmark results.
- No claim is made about full models, graphs, IPC, distributed execution or third-party extensions.

- stock_standard: PASS (3 assertions groups)
- stock_streams: PASS (1 assertions groups)
- off_standard: PASS (3 assertions groups)
- off_streams: PASS (1 assertions groups)
- off_frozen: PASS (1 assertions groups)
- host_standard: PASS (4 assertions groups)
- host_streams: PASS (1 assertions groups)
- host_uncached: PASS (4 assertions groups)
- host_oom: PASS (1 assertions groups)
- host_graph_reject: PASS (1 assertions groups)
- host_fraction_reject: PASS (1 assertions groups)
- host_frozen: PASS (4 assertions groups)
- host_bad_flag: PASS (1 assertions groups)
- host_async_reject: PASS (1 assertions groups)
- host_expandable_reject: PASS (1 assertions groups)
- host_reconfigure_reject: PASS (1 assertions groups)
