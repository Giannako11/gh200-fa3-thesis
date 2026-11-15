# Step 3: selected CUDA tensors in Grace DRAM

The runner builds a small C++ shared library and loads its `grace_alloc` and
`grace_free` callbacks in the stock PyTorch 2.11 environment built in step 2.
`GracePool.empty_on_grace` scopes `torch.cuda.use_mem_pool` to a single
`torch.empty` call. The allocated storage remains in the pool after leaving
the context. Other CUDA allocations use the stock allocator.

The callbacks reuse the step 2 allocation approach: page-align a range
inside a system `malloc` allocation, bind it to NUMA node 0 before first
touch, and query its page locations with `move_pages`. The library caps
outstanding host memory at 128 MiB, including page alignment overhead.
The free callback synchronizes the device, resets the range policy, and
frees the original pointer. The runner disables core dumps and retains
worker errors in the log.

## Run on Grace2

From the repository root on an idle GH200, with the step 2 build available:

```bash
bash thesis/scripts/run_pytorch_custom_alloc.sh --plan
bash thesis/scripts/run_pytorch_custom_alloc.sh
```

The default build root is `/local/$USER/thesis_host_malloc` when available,
with `$HOME/thesis_host_malloc` as the fallback. If you used a different
build root, pass `--root /absolute/path`. The runner uses `stock_env/bin/python`
and the CUDA Toolkit from the step 2 manifest. Use `--cuda-home` to override
the toolkit path. The runner does not install packages or rebuild PyTorch.

## What the run checks

1. A regular 32 MiB CUDA tensor consumes 32 MiB HBM and has a CUDA device
   pointer. A second 32 MiB CUDA tensor resides on Grace node 0 with no HBM
   growth. A third tensor, allocated outside the pool for the result, is
   again in HBM.
2. Both inputs pass the same in-place transform. A single
   `torch.add(hbm_input, grace_input, out=hbm_output)` produces the expected
   value in every output element. The Grace pages are verified again.
3. After synchronizing, dropping the Grace tensor, and closing the pool,
   the custom allocator reports zero live bytes. `worker.log` records the
   matching alloc/free callbacks.

Each run writes `manifest.json`, `compile.log`, `worker.log`, `result.json`
and `status.json` under `thesis/outputs/torch_custom_alloc_*`. A failure inside
a native callback can end the worker before it writes `result.json`. In that
case, check `status.json` and `worker.log`.

This is a functional test in eager mode on one GH200. The attention workload
and measurements of the effect of placing sparse buffers belong to the
follow-up experiment.
