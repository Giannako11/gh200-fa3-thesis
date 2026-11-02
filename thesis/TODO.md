# this is my todo file where i keep track of some things so i dont forget anything

// 14.10.26 -- setup qwen etc ----- done!

## 01.11.26 — Next steps

1. Benchmark regular GPU loads against TMA with increasing input sizes.
   Compare the times, find where each approach works better and profile the
   main differences. Keep total input size separate from tile size and check
   the effect of caches. Fix the `mbind` argument before the new runs.

2. Check how to replace GPU-memory allocations with normal host allocations
   (`malloc`) in PyTorch, so selected tensors can use Grace DRAM instead of
   Hopper HBM while still being used by GPU operations.
   Verify where the pages are placed and that the GPU can access them correctly.

3. Look into custom allocators in PyTorch: how to provide our own allocation
   and free functions, and how to choose which tensors use them.
   Start with a small test before trying it with a full model.
