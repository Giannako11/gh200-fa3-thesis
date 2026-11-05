# Controlled GPU cache conditions and independent allocations

Run from the repository root on an idle GH200:

```bash
conda activate fa3
bash thesis/scripts/run_tma_controlled_sweep.sh
```

For a long run that survives an SSH disconnect:

```bash
nohup bash thesis/scripts/run_tma_controlled_sweep.sh > tma_controlled.log 2>&1 &
tail -f tma_controlled.log
```

`Ctrl-C` exits `tail`; the background sweep continues. A server shutdown, resource
reclamation or job scheduler limit can still terminate it. The wrapper activates
`fa3` when necessary using `$HOME/miniconda3/etc/profile.d/conda.sh`.

The runner compiles its archived generated CUDA source, runs a mandatory cache
conditioning probe, then starts the sweep. No source or settings from earlier
outputs are modified. Existing sampled results remain useful under their original
warmed, interleaved measurement protocol.

## Defaults and scope

The original 145 valid `(input size, tile)` pairs are retained: powers of two
from 4 KiB to 16 GiB and seven tiles, subject to whole-tile divisibility.
The sources are `hbm`, `pinned`, `registered`, and `malloc_bound`. The last uses
page-aligned `posix_memalign` plus allocation-range `mbind(MPOL_BIND)` before
initialization, without CUDA registration. Plain migration-prone `malloc` is
not substituted for it. All workers run with CPU/NUMA binding to node 0.

| Quantity | Default |
|---|---:|
| Allocation configurations | 580 |
| Cache conditions | 2 |
| Independent processes/allocations per condition/config | 3 |
| Total worker processes | 3,480 |
| Kernels per worker | 3 |
| Initial warmup launches per kernel | 10 |
| Timed samples per kernel/replicate | 30 |
| Raw timing samples | 313,200 |
| Pooled samples per kernel/config/condition | 90 |
| Pooled result rows per condition | 1,740 |
| PNG plots per condition, full grid | 39 |

This is substantially more work than the earlier single-allocation sweep.
Progress reports completed workers, the current condition/configuration, and
elapsed time. There is no assumed run-time estimate.

## What each sample measures

Two conditions are reported in separate folders and CSVs:

* `gpu_evicted`: GPU scratch eviction -> timed target kernel.
* `self_primed`: the SAME eviction -> one untimed launch of the SAME target
  kernel -> timed target kernel. The primer is additional to the 10 initial
  warmups and runs before every timed sample.

Each kernel has its own block of warmup and samples; other benchmark kernels do
not alternate between those samples. Across the three fresh allocations, block
order rotates through all three positions. Condition order alternates across
configs/replicates. At larger inputs, one primer does not imply the entire
input fits in cache; it defines the immediately preceding workload.

CUDA events bracket only the complete copy kernel, including its HBM output
writes. Scratch traffic, coverage queries, allocation, initialization, warmup,
the optional primer, page queries, and correctness verification are excluded.
The three target GPU kernels and their tiling are preserved from `tma_sweep.cu`.
No clocks, driver settings, migration controls or global kernel settings are
changed.

## Eviction and its limits

The eviction kernel uses explicit `.ca` global loads and `.wb` global stores to
a separate HBM allocation. It makes two passes, with at least eight CTAs per
reported SM, at least 512 KiB per CTA (also at least twice reported shared memory
per SM), and at least eight times reported L2 capacity overall. Each conditioning
call records visited SM IDs, retries up to three times, and fails if coverage is
incomplete. Scratch writes prevent the compiler from discarding the traffic.
The traffic is outside the target timing interval.

Before every invocation, including resume, a paired probe uses a copy workload
whose combined input/output footprint is half the reported L2. Twelve samples
per condition alternate condition order after ten warmup cycles. All output
values are verified. The evicted median must exceed the primed median by more
than 5%, and the evicted 25th percentile must exceed the primed 75th percentile.
Otherwise the full sweep stops, retaining probe JSON/logs for inspection. This
is an empirical sanity check, not a physical cache-state measurement. A failed
probe can be inconclusive even if the eviction code works; inspect it before
changing thresholds or rerunning.

**This is GPU eviction conditioning, not a portable guaranteed hardware cache
invalidation.** SM coverage and footprint checks cannot prove that every cache
line is absent. It does not flush Grace CPU caches, TLBs or all interconnect
caches, nor guarantee service from LPDDR/HBM on every access. In particular,
small host inputs may still be served through CPU-side caches. Do not label
these numbers “cold LPDDR bandwidth.” `cudaCtxResetPersistingL2Cache` is not
used as a full cache flush. Effective input GB/s counts input bytes once and is
not measured physical DRAM/NVLink traffic. Use profiler hardware counters for
claims about physical traffic, separately from these timing runs.

## Correctness, placement and statistical reporting

Every kernel is verified against the expected full output before warmup and
after its sample block. Host page placement is checked before warmup, after
warmup, after samples before verification, and after final verification.
Every snapshot must show all pages on CPU node 0, without unknown/error pages.
For `malloc_bound`, the VMA policy is also checked explicitly. The runner fails
on nonzero worker exits, incorrect outputs, placement failures, invalid samples
or incomplete conditioning. Snapshots do not prove continuous residency.

Per-condition `results.csv` reports the median of all 90 raw samples, with pooled
min/max, p25/p75/p95, mean, sample SD and CV. This is a pooled median, not a
median of replicate medians. Equal numbers of samples give each allocation
equal weight. Per-replicate statistics remain in `replicate_results.csv`;
replicate median min/max and CV are also included in pooled rows. Pooled page
counts are sums across replicates; warmup count is per replicate, and pooled
`first_sample_position=-1` means there is no single block position. Each raw
sample retains its replicate, local sample index and block position.

No outliers are removed. Summaries count pooled sample CV >10% and replicate
median CV >5% as descriptive diagnostics. A correctness/placement PASS is not a
claim of low noise, confidence intervals, independent machine conditions, or
statistical significance. Plotted error bars show pooled IQR, not a confidence
interval. Fresh allocations are independent allocation replicates on the same
machine; thermal state and other system effects can remain correlated.

One-second `nvidia-smi` telemetry records clocks, temperature, power, utilization
and memory usage. Per-worker UTC timestamps support alignment with those logs.
Inspect clock/thermal changes and replicate variability before drawing small
performance distinctions. Telemetry does not lock clocks; early telemetry
failure is reported and saved. Other GPU processes are checked before start;
the runner cannot reserve the GPU against jobs started later.

## Outputs, interruption and resume

The printed `thesis/outputs/tma_controlled_sweep_*` folder contains:

* Archived CUDA source and generator dependencies, build/environment logs,
  compiler/GPU/protocol manifest and original base commit.
* Timestamped `cache_check_*.json`/logs and `telemetry_*.csv`/status.
* `cases/<condition>/repNN/<config>/` with rows, raw samples, worker log,
  command/timestamps, conditioning metadata and hashed completion checkpoint.
* `<condition>/results.csv`, `replicate_results.csv`, `samples.csv`, `summary.md`.
* `plots/<condition>/` with PNGs and plot checks/notes; no PDFs.
* Overall `summary.md`, `status.json` and `failure.txt` when interrupted/failed.
* A sibling `.tar.gz` diagnostic bundle, including source/raw data/plots and
  failed attempts, excluding the executable. The runner prints `SEND_THIS_FILE`.

Resume with the exact original sampling arguments (defaults shown):

```bash
bash thesis/scripts/run_tma_controlled_sweep.sh --resume /absolute/path/to/tma_controlled_sweep_FOLDER
```

Completed workers are hash-checked and semantically validated before reuse.
Aggregate exports are reconstructed from authoritative case files, so interrupted
writes cannot duplicate/drop samples. Incomplete attempts are retained as
`.failed_*` folders. Resume rejects source, protocol, GPU/driver or compiler
changes, reruns the cache probe, and continues missing workers. The output lock
also prevents overlap with the older sweep runners. Normal signals/timeouts
retain diagnostics; a hard kill/power loss may require resume to rebuild exports.

For inspection or a short actual-GPU trial:

```bash
python thesis/scripts/run_tma_controlled_sweep.py --plan
bash thesis/scripts/run_tma_controlled_sweep.sh --smoke
```

Smoke retains both conditions, all four sources, all three replicates and full
default sampling, using just 1 MiB / 32x32. It is not the full thesis dataset.
`--warmup`, `--samples` and `--replicates` are configurable; samples must be at
least six and divisible by six, and replicates at least three and divisible by
three to balance block positions. `--skip-plots` and `--no-telemetry` are explicit
diagnostic options. The conditioning probe cannot be bypassed.

## Local verification boundary

CPU tests cover strict sample/statistic/placement/conditioning validation,
pooled aggregation, unchanged target-kernel generation, and inconclusive probes.
An end-to-end local fixture also exercises the native C++ statistics/CSV writer,
worker orchestration, fail-stop, recovery, exports, archive and no-replay resume.
These do not execute CUDA. Actual CUDA compilation and GPU conditioning must
be verified on the GH200; the runner does both before the expensive sweep.

Protocol references:

* [NVIDIA PTX ISA: cache operators](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#cache-operators):
  the load/store cache operators are performance hints.
* [NVIDIA Nsight Compute: cache control](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#cache-control):
  profiling can use a distinct replay/cache-control regime. Collect traffic
  counters separately and record its cache/clock settings; do not silently
  mix profiled timings with this event-timed dataset.
