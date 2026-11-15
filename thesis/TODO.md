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

2. Verify that PyTorch CUDA tensors can use Grace DRAM via system `malloc`
   and still execute ordinary GPU operations. ----- `completed` (07.11.26)

   Built stock and patched PyTorch from the same pinned source in separate
   prefixes outside the repository. The patched native CUDACachingAllocator
   uses the original CUDA path by default. The runner sets
   `THESIS_HOST_MALLOC=1` only for its host-test subprocesses.

   GH200 validation passed 16/16 checks on one GPU in eager mode. A 32 MiB
   allocation increased HBM use by 32 MiB in stock and patched OFF. With
   the flag ON, cached and uncached tests showed no increase in HBM use
   and confirmed all 512 pages on CPU NUMA node 0. GPU operations on the
   host-backed tensor returned correct results. Stream lifetime, release,
   and bounded OOM recovery also passed.

   See [GH200 results](outputs/torch_host_malloc_20261107T171406Z_o7jc7yft/summary.md)
   and [implementation notes](pytorch_host_malloc/README.md).

3. Add a custom allocator for selected CUDA tensors in PyTorch. ----- `in progress` (15.11.26)

   Plan: compile a small shared library with Grace DRAM allocation and free
   callbacks. Load it in stock PyTorch 2.11 with `CUDAPluggableAllocator` and
   `MemPool`. Use the pool only when creating a selected tensor.

   GH200 checks: create a CUDA tensor in Hopper HBM and one in Grace DRAM
   in the same process, verify both placements, run a GPU operation that
   uses both tensors, and confirm that pool teardown calls the custom free
   callback and releases its allocations.

   Follow-up after validation: try selected sparse attention buffers in
   Grace DRAM with dense data in Hopper HBM on a small workload.
