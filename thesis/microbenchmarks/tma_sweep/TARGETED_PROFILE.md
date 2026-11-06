# Targeted GH200 TMA profiling

From the repository root on Grace2, with the completed controlled sweep still
in `thesis/outputs/`:

```bash
git pull --ff-only
bash thesis/scripts/run_tma_targeted_profile.sh --plan
nohup bash thesis/scripts/run_tma_targeted_profile.sh > tma_targeted_profile.log 2>&1 &
tail -f tma_targeted_profile.log
```

The script selects the most recently completed `tma_controlled_sweep_*` by
default. To choose a particular run, add `--sweep /absolute/path/to/run` to
both the plan and actual invocation. The archived `controlled` executable and
its matching `controlled.cu` must be present. The script does not recompile or
repeat the whole sweep.

Default: `registered`, 32 MiB, 128x128, `tma_shared`; three independent
processes/allocations per protocol and two timed sample positions (3 and 5)
per process. This produces 12 `.ncu-rep` reports. Add
`--sources registered pinned malloc_bound` to cover the other two host
sources, or `--replicates 1 --sample-indices 3` for a quick two-report check.
The extra sources need more profiling time and disk space.

Nsight Compute filters on `tma_shared`, skips the correctness and ten warmup
TMA launches, and in `self_primed` additionally skips the untimed same-input
primer. It captures exactly one timed TMA launch per application run. The
original executable performs its complete scratch eviction and coverage checks
before that launch. Application replay, `--cache-control none`, and
`--clock-control none` retain the original conditioning sequence across
counter collection passes. The script verifies output, CPU page placement,
conditioning metadata, and the original source hash.

The printed `thesis/outputs/tma_targeted_profile_*` folder holds `plan.json`,
`status.json`, and one folder per profile with `tma.ncu-rep`, `ncu_raw.csv`,
`ncu.log`, `command.json`, `rows.csv`, `samples.csv`, and `conditioning.json`.
`status.json` must show 12 completed of 12 expected for the default run.
Inspect error logs if a case stops, including possible NVIDIA GPU counter
permission errors. Do not use CUDA event times from profiled runs as benchmark
results or infer the 99/125 microsecond split from them; compare cache and
memory counters and use the earlier unprofiled samples for timing.
