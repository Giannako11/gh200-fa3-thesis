# Step 2: opt-in system malloc inside PyTorch on GH200

The task is to replace the native PyTorch CUDA allocator's underlying
`cudaMalloc` allocations with ordinary system `malloc` on Grace DRAM, while
ordinary CUDA tensor operations still execute on the GPU. This package patches
PyTorch itself; it does not wrap a manually allocated tensor with `from_blob`.
Selective allocation policies/custom user MemPools belong to step 3.

The implementation is prepared for Grace2. **A successful local CPU check is
not a successful GH200 run.** The 16-case GPU validation must pass before
claiming step 2 is verified on this machine. Build and runtime failures retain
their diagnostics and stop instead of silently using a different allocation.

## Isolation and default behavior

- The existing `fa3` installation is only inspected, never a pip/conda install
  destination. Three fresh conda prefixes live under the external build root:
  `build_env`, `stock_env`, and `fa3_host_malloc`.
- `stock_env` has unmodified upstream PyTorch. `fa3_host_malloc` has the patched
  build, whose default is OFF. Both wheels use the same upstream commit and
  build flags. The build environment is separate from both runtime prefixes.
- The script sets `THESIS_HOST_MALLOC=1` only in host-test subprocesses. It does
  not edit shell startup files, globally export variables, change the default
  environment, or change system migration/clock settings.
- The flag is frozen when libtorch loads. Changing it after import cannot mix
  two allocation families. Only unset/`0` and `1` are valid.
- OFF retains the native CUDA allocation/free/copy paths. ON redirects actual
  backing blocks, including the uncached path. PyTorch's native block splitting,
  reuse, stream bookkeeping and tensor CUDA dispatch remain in place.

## Exact source and files

Upstream: `pytorch/pytorch` tag `v2.11.0`, commit
`70d99e998b4955e0049d13a98d77ae1b14db1f45`.
The audited `CUDACachingAllocator.cpp` Git blob is
`2ab4effc7853dc0ce62ebc60e6dbb8ced9b72cca`.
This is a pinned experimental build, not an assumption that the installed fa3
wheel has the identical source commit or build options.

`patches/native_host_malloc.patch` changes
`c10/cuda/CUDACachingAllocator.cpp` and adds
`c10/cuda/ThesisHostMalloc.h` in the external PyTorch checkout. The standalone
header in this directory is the reviewable source of the header in the patch;
local tests check that the two are byte-identical.

Scripts under `thesis/scripts/` provide inspection, build, execution and common
utilities. `thesis/tests/test_torch_host_malloc.py` is the GH200 subprocess test
worker; `test_torch_host_malloc_local.py` exercises the helper with CUDA/NUMA
test doubles on CPU. Those doubles are never compiled into the PyTorch build.

## Run on Grace2

From the repository root, activate the original fa3 environment:

```bash
cd ~/repos/gh200-fa3-thesis
git pull --ff-only
conda activate fa3
python thesis/scripts/inspect_torch_host_malloc.py
bash thesis/scripts/build_torch_host_malloc.sh --plan
```

The default external root is `/local/$USER/thesis_host_malloc` when
`/local/$USER` exists, otherwise `$HOME/thesis_host_malloc`. It must be outside
the repository and outside the active Python environment. Both scripts accept
`--root /absolute/path` when a different location is needed; use the same root
for build and execution. An existing nonempty, unowned or differently configured
root is rejected. The build never resets a source tree with unrelated edits.

The build downloads source/submodules and dependencies from GitHub, conda-forge
and PyPI. It requires a working C++ compiler, the CUDA Toolkit (12.8.1 on the
current host), network access and at least 80 GiB free in the build filesystem.
Default parallelism is four build jobs; this is adjustable with `--jobs`.
Its duration depends on downloads and compilation; it is not a short GPU test.

```bash
nohup bash thesis/scripts/build_torch_host_malloc.sh > torch_host_malloc_build.log 2>&1 &
tail -f torch_host_malloc_build.log
```

The outer log shows stages. Full compiler output, including Ninja progress,
is in `<root>/build.log`. `Ctrl+C` stops `tail`, not the nohup build.
The build creates an unmodified wheel first and then applies the patch and
builds the second wheel. CUDA capability is 9.0. This bounded eager-inference
build enables CUDA/cuBLAS and Kineto, and disables cuDNN, distributed/NCCL/MPI,
MAGMA and MKLDNN. All exact flags and wheel hashes are recorded in
`<root>/manifest.json`. It does not install FlashAttention or a full LLM stack.

After `build_status.json` reports `state: complete`, run on an idle GH200:

```bash
bash thesis/scripts/run_torch_host_malloc.sh --plan
nohup bash thesis/scripts/run_torch_host_malloc.sh > torch_host_malloc_run.log 2>&1 &
tail -f torch_host_malloc_run.log
```

The full run has 16 checks, with a progress bar, independent processes, CPU and
memory binding to node 0, a shared lock with the TMA sweep, and GPU telemetry.
Each worker has a five-minute limit and is terminated on timeout/interruption.
No resumption reuses a failed runtime directory; a new run makes a fresh folder.
Use `--case host_standard` for an individual diagnostic, not as a substitute for
the full validation. There is no automatic Git push in these scripts.

## Allocation and release contract

1. The host mode requires exactly one visible GH200, ATS/pageable/concurrent
   access, the native backend, and no expandable segments. It rejects graph
   capture, CUDA IPC, user MemPools, cross-device copies and HBM-based memory
   fractions. Those features are outside this step's supported scope.
2. `std::malloc` overallocates enough to expose an interior system-page-aligned
   range. The original malloc pointer is recorded and is the one freed. Complete
   pages inside that allocation receive per-range `MPOL_BIND` to node 0.
3. CPU first touch populates the range; `get_mempolicy` checks the full node mask
   and `move_pages` queries every page before any tensor can use it. No CUDA
   registration, `cudaMallocManaged`, host pinning or HBM fallback is used.
4. Actual host allocation charges, including alignment overhead, are bounded.
   The helper's standalone default is 1 GiB; the validation runner explicitly
   uses 256 MiB. `THESIS_HOST_MALLOC_MAX_BYTES` is the process's positive byte
   cap. It is an allocation bound, not a guarantee against system memory pressure.
5. Blocks returned to libc are device-synchronized and rechecked for physical
   residency. The VMA policy is reset before `free(base)`. On a synchronization,
   ownership or placement failure, the helper refuses to free that block.
   Process exit still allows the OS to reclaim it. Ordinary tensor deletion can
   retain blocks in PyTorch's cache; actual releases are logged separately.
6. CUDA-to-CUDA copies within this host-backed allocator use a synchronized CPU
   `memmove`, as system-to-system CUDA copy APIs should be avoided on ATS.
   This is logged as `host_copy`. It is not presented as GPU copy performance.
   Arithmetic, reductions and matmul are ordinary CUDA-dispatched operations.

The absolute `THESIS_HOST_MALLOC_LOG` path must be fresh. It is opened
exclusively, never truncated, and receives JSONL configuration, allocations,
budget rejections, host copies and frees. Pointer membership and matching
allocation/free families are checked. No host path is entered or log opened
when the flag is OFF. Allocation first touch/page queries and release
synchronization add overhead: this package establishes correctness and placement,
not an optimized allocator or publishable latency measurement.

## What constitutes PASS

`thesis/outputs/torch_host_malloc_<timestamp>_<suffix>/` contains:

- `manifest.json`: upstream commit, patch/header/script hashes and wheel hashes.
- `status.json` and `summary.md`: 16/16 checks and exit code 0 for the default run.
- Per-case `command.json`, `worker.log`, `result.json`, and host allocation JSONL.
- Placement evidence for actual tensor storages, including before/after GPU
  writes; synchronized CPU reads witness the GPU's result at the same pointer.
- A physical HBM observation for a 32 MiB tensor after warmup, separate from
  logical allocator statistics, and release checks after `empty_cache()`.
- FP32/FP16/BF16/integer operations, CPU/CUDA and CUDA/CUDA copies, views,
  reductions, cuBLAS matmul, cross-stream lifetime stress, uncached allocations,
  bounded/recoverable OOM, frozen mode, and unsupported-configuration rejection.
- `environment.txt`, `telemetry.csv` and exact copies of the patch/header.

All 16 checks must pass. Merely seeing `tensor.device == 'cuda'`, a successful
build, or a successful hand-made CUDA kernel does not satisfy this gate.
Physical placement is verified at snapshots, not continuously. Logical
`torch.cuda.memory_allocated/reserved` count allocator storage and must not be
relabelled as physical HBM. CUDA context/library allocations can still use HBM.
The numerical checks use CPU references; their wall times are diagnostics.
The direct-pointer witness plus allocation/page evidence establishes the tested
storage path, not a universal absence of temporaries in arbitrary library code.

## Failure and repeatability

Read the first failing worker log and keep the output folder. `mbind` or
`move_pages` permission errors are failures, not a reason to skip residency
checks. Source, header or patch mismatches require a fresh matching build root.
Build retries reuse only hash-verified complete wheels; partial wheel artifacts
or unrelated source changes are not overwritten. There is no destructive cleanup
command. The existing fa3 environment remains the independent original install.

## Local checks (no GPU required)

```bash
python -m unittest discover -s thesis/tests -p test_torch_host_malloc_local.py -v
```

References: [PyTorch 2.11 CUDA semantics](https://docs.pytorch.org/docs/2.11/notes/cuda.html),
[pinned allocator source](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/c10/cuda/CUDACachingAllocator.cpp),
[NVIDIA unified memory and ATS](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html).
