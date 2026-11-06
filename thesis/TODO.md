# this is my todo file where i keep track of some things so i dont forget anything

## 14.10.26 -- setup qwen etc ----- done!

## 01.11.26 — Next steps

1. Benchmark regular GPU loads against TMA with increasing input sizes. ----- `completed*` (06.11.26)
   Compare the times, find where each approach works better and profile the
   main differences. Keep total input size separate from tile size and check
   the effect of caches. Fix the `mbind` argument before the new runs.

   Completed: full controlled sweep, corrected per-range host binding and
   CPU page placement checks, independent allocations, repeated samples,
   separate GPU-evicted/self-primed protocols, plots and targeted profiling.
   The controlled sweep passed 3,480/3,480 workers and 313,200/313,200 samples.

   `*` Follow-up paused: the two timing clusters around 99/125 us for host TMA
   at 32 MiB remain unexplained. The 12/12 targeted profiles passed and show
   different GPU HBM traffic between protocols, but the collected counters
   do not isolate Grace-Hopper C2C traffic or input reads from output writes.
   Remaining work: identify the cause with targeted system-memory/C2C counters
   and timing observations with lower profiler overhead. Keep the original
   unprofiled sweep as the timing reference; profiler CUDA-event times are
   heavily perturbed. This does not block starting step 2; no further full
   sweep is currently scheduled.

2. Check how to replace GPU-memory allocations with normal host allocations
   (`malloc`) in PyTorch, so selected tensors can use Grace DRAM instead of
   Hopper HBM while still being used by GPU operations. ----- `in progress` (06.11.26)
   Verify where the pages are placed and that the GPU can access them correctly.
   Implement the change in PyTorch's native CUDACachingAllocator, behind an
   OFF-by-default process flag. Keep fa3 untouched; build stock and patched
   PyTorch in separate prefixes outside the repository. The new host-malloc
   runner enables it only for its own subprocesses. This is not a from_blob
   wrapper or a step-3 custom user allocator.
   Implementation/scripts and CPU-only helper checks are prepared under
   pytorch_host_malloc/. Real source builds and all 16 GH200 runtime checks
   remain required before marking step 2 completed. Check ordinary CUDA tensor
   operations, pointer provenance, node-0 placement, bounded OOM, stream lifetime,
   release, and original/OFF behavior. See pytorch_host_malloc/README.md.

3. Look into custom allocators in PyTorch: how to provide our own allocation
   and free functions, and how to choose which tensors use them.
   Start with a small test before trying it with a full model.
