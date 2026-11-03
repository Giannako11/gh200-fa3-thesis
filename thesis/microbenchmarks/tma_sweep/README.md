# TMA sweep on GH200

We compare three GPU copy kernels: regular loads directly to HBM output,
regular loads through GPU shared memory, and TMA through GPU shared memory.
The input is HBM, CUDA-allocated mapped host memory, registered host memory,
or plain `malloc` memory.
Each GPU block handles one contiguous tile. The TMA view has tile-width rows;
this is a packed-tile copy test, not an attention or matrix multiplication test.

Run from a free GPU slot, inside tmux:

```bash
bash thesis/scripts/run_tma_sweep.sh --plan
bash thesis/scripts/run_tma_sweep.sh
```

Defaults: 23 input sizes from 4 KiB to 16 GiB, seven tiles, 128 GPU threads
per GPU block. Sizes smaller than a tile are skipped: 580 allocation groups,
1,740 kernel measurements. There is one timed launch per kernel, no warmup,
no cache flushing and no automatic repetitions. Kernel order rotates by group.
Earlier copies and output checks can affect the cache state of later kernels.

Sources: `hbm,pinned,registered,malloc`. The `malloc` case uses `std::malloc`
and `std::free`, without registration, pinning calls, NUMA binding or input copies
to HBM. It requires GPU system-memory access with ATS, prints the device attributes
and passes the original host pointer to the GPU kernels. CPU writes initialize
the input; page queries check its location before and after each timed kernel.
This does not impose a placement policy or prove that pages never moved during
the kernel. The output is always HBM. If TMA descriptor encoding fails, its row
records an error while the ordinary-load kernels are still tested.

Parameters are at the top of the Bash file and can also be passed as environment
variables. For a pilot with every tile that fits:

```bash
SIZES=4KiB,64MiB,1GiB,16GiB bash thesis/scripts/run_tma_sweep.sh
```

Results go under `thesis/outputs/tma_sweep_<timestamp>_<suffix>/`.
`results.csv` contains the timing, status, launch order and page-location checks.
`summary.md` counts correct, failed and missing results. Each group has its own log.
The progress bar counts completed groups, including saved groups when resuming.
Failed groups also count as completed; check the summary for correctness.
`ok` means all output values matched. CPU page placement is a separate column:
1 = verified before and after, 0 = not verified, -1 = HBM case.
`mbind_success=-1` means mbind was not requested. Effective input GB/s is input
bytes divided by the GPU event interval, including output stores; it is not a
measurement of physical DRAM or NVLink bandwidth. These single-launch times
include first-use effects and are not steady-state latency measurements.

Resume with the same parameters and unchanged source files/compiler/GPU identity:

```bash
bash thesis/scripts/run_tma_sweep.sh --resume thesis/outputs/RUN_FOLDER
```

Finished groups, including recorded failures, are kept. Incomplete groups rerun.
Timeouts, signals and serious CUDA faults stop the sweep. The initial GPU process
check does not reserve the machine. Profiling and plots are separate later steps.

Draft status: CUDA compilation and GH200 execution still need validation.
