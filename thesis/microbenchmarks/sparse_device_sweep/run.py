"""Run or resume the two density endpoints for the large GH200 shapes."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter

from grid import DEVICES, SHAPES, cases

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
WORKER = HERE / "worker.py"
OUTPUTS = REPO / "thesis/outputs"


def save_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def write_csv(path, rows, fields):
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def workload(smoke):
    return cases(((64, 80, 48),)) if smoke else cases()


def environment(threads):
    env = os.environ.copy()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(threads)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def progress(done, total, text):
    width = 24
    fill = width * done // total
    print(f"[{'#' * fill}{'-' * (width - fill)}] {done}/{total} {text}", flush=True)


def read_events(path):
    if not path.exists():
        return []
    events = []
    for line in path.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # A process may be interrupted during its final write.
            continue
    return events


def execute(args, case, device, out, env):
    folder = out / "cases" / case.name
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / f"{device}.json"
    steps = folder / f"{device}.steps.jsonl"
    if final.exists():
        prior = json.loads(final.read_text())
        if prior["status"] == "pass" or not args.retry_failed:
            return prior
    if final.exists() or steps.exists():
        # Preserve failed and interrupted attempts before trying again.
        history = Path(tempfile.mkdtemp(prefix=f"{device}_attempt_", dir=folder))
        for name in (final, steps, folder / f"{device}.log"):
            if name.exists():
                name.replace(history / name.name)

    values = (case.m, case.k, case.n, case.a_pct, case.b_pct)
    command = ["numactl", "--cpunodebind=0", "--membind=0", sys.executable, str(WORKER),
               "--case", *map(str, values), "--device", device, "--threads", str(args.threads),
               "--warmups", str(args.warmups), "--repeats", str(args.repeats), "--steps", str(steps)]
    start = perf_counter()
    status = "failed"
    error = ""
    with (folder / f"{device}.log").open("w") as log:
        try:
            process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=args.timeout, check=False)
            if process.returncode == 0:
                status = "pass"
            else:
                error = f"worker exit {process.returncode}"
        except subprocess.TimeoutExpired:
            status = "timeout"
            error = f"exceeded {args.timeout:g}s"
        except OSError as exc:
            error = str(exc)

    wall = perf_counter() - start
    events = read_events(steps)
    if status == "pass" and not any(e["event"] == "pass" for e in events):
        status, error = "failed", "worker did not record validation PASS"
    started = [e for e in events if e["event"] == "start"]
    result = {
        "config": case.name, "m": case.m, "k": case.k, "n": case.n, "a_pct": case.a_pct,
        "b_pct": case.b_pct, "strategy": case.strategy, "device": device, "status": status,
        "wall_seconds": wall, "last_stage": started[-1]["stage"] if started else "", "error": error,
    }
    samples = defaultdict(list)
    for e in events:
        if e.get("sample") is not None and e["sample"] > 0 and e["event"] in {"done", "device_time"}:
            key = "cuda_compute" if e["event"] == "device_time" else e["stage"]
            samples[key].append(e["seconds"])
    for field in ("prepare_b", "compute", "cuda_compute", "materialize_output",
                  "transfer_hbm_to_grace", "transfer_grace_to_hbm", "end_to_end"):
        result[f"median_{field}_seconds"] = statistics.median(samples[field]) if samples[field] else ""
    save_json(final, result)
    return result


def summarize(out, configs):
    devices = []
    stage_rows = []
    for case in configs:
        for device in DEVICES:
            prefix = out / "cases" / case.name / device
            result_path = prefix.with_suffix(".json")
            if not result_path.exists():
                continue
            devices.append(json.loads(result_path.read_text()))
            for event in read_events(prefix.with_suffix(".steps.jsonl")):
                stage_rows.append({"config": case.name, **event})
    if devices:
        write_csv(out / "device_times.csv", devices, list(devices[0]))
    if stage_rows:
        write_csv(out / "steps.csv", stage_rows,
                  ["config", "device", "sample", "event", "stage", "seconds"])

    by_case = defaultdict(dict)
    for row in devices:
        by_case[row["config"]][row["device"]] = row
    config_rows = []
    for case in configs:
        records = by_case.get(case.name, {})
        if not records:
            continue
        statuses = [records.get(device, {}).get("status", "pending") for device in DEVICES]
        config_rows.append({"config": case.name, "m": case.m, "k": case.k, "n": case.n,
                            "a_pct": case.a_pct, "b_pct": case.b_pct, "strategy": case.strategy,
                            "grace_status": statuses[0], "hopper_status": statuses[1],
                            "wall_seconds": sum(r["wall_seconds"] for r in records.values())})
    if config_rows:
        write_csv(out / "config_times.csv", config_rows, list(config_rows[0]))

    counts = Counter(r["status"] for r in devices)
    lines = ["# GH200 large-shape density endpoints", "",
             f"- Passed device runs: {counts['pass']} / {2 * len(configs)}.",
             f"- Failed: {counts['failed']}. Timed out: {counts['timeout']}.",
             f"- Recorded device worker wall time: {sum(r['wall_seconds'] for r in devices):.1f} s.", ""]
    lines += ["| M×K×N | Density A / B | Grace wall (s) | Hopper wall (s) |",
              "|---|---:|---:|---:|"]
    for case in configs:
        elapsed = []
        for device in DEVICES:
            row = by_case[case.name].get(device)
            if row is None:
                elapsed.append("pending")
            elif row["status"] != "pass":
                elapsed.append(f"{row['status']} ({row['wall_seconds']:.2f} s)")
            else:
                elapsed.append(f"{row['wall_seconds']:.2f}")
        lines.append(f"| {case.m}×{case.k}×{case.n} | {case.a_pct}% / {case.b_pct}% | "
                     f"{elapsed[0]} | {elapsed[1]} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    return counts["pass"]


def manifest(args):
    return {
        "smoke": args.smoke, "warmups": args.warmups, "repeats": args.repeats,
        "threads": args.threads, "python": sys.executable,
        "shapes": [list(shape) for shape in SHAPES],
        "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (Path(__file__), WORKER, HERE / "grid.py")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="list configs without starting CUDA")
    parser.add_argument("--smoke", action="store_true", help="test both density endpoints at a small shape")
    parser.add_argument("--resume", type=Path, help="continue an existing run with the same settings")
    parser.add_argument("--retry-failed", action="store_true", help="retry failures when resuming")
    parser.add_argument("--timeout", type=float, default=1200, help="seconds per device worker")
    parser.add_argument("--threads", type=int, default=16, help="CPU and BLAS thread budget")
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1 or args.warmups < 0 or args.repeats < 1 or args.timeout <= 0:
        parser.error("threads, repeats, and timeout must be positive and warmups nonnegative")
    if args.retry_failed and args.resume is None:
        parser.error("--retry-failed requires --resume")
    configs = workload(args.smoke)
    if args.plan:
        for case in configs:
            print(f"{case.name} {case.strategy}")
        print(f"{len(configs)} configs, {2 * len(configs)} device workers, "
              f"{2 * len(configs) * args.repeats} timed iterations")
        return 0

    for cmd in ("numactl", "nvidia-smi"):
        if not shutil.which(cmd):
            parser.error(f"missing command: {cmd}")
    env = environment(args.threads)
    info = subprocess.check_output(
        ["numactl", "--cpunodebind=0", "--membind=0", sys.executable, "-c",
         "import json,numpy,scipy,torch; "
         "assert torch.cuda.is_available(), 'CUDA unavailable'; "
         "assert 'GH200' in torch.cuda.get_device_name(0), 'expected GH200'; "
         "print(json.dumps(dict(numpy=numpy.__version__,scipy=scipy.__version__,"
         "torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0))))"],
        text=True, env=env, timeout=60,
    )
    identity = {**manifest(args), "environment": json.loads(info)}
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    with (OUTPUTS / ".tma_sweep.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pids = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
        if any(line.strip().isdigit() for line in pids.splitlines()):
            parser.error("GPU compute process present; run on an idle GH200")
        if args.resume:
            out = args.resume.expanduser().resolve(strict=True)
            if json.loads((out / "manifest.json").read_text())["identity"] != identity:
                parser.error("resume needs the original code, environment, and run settings")
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out = Path(tempfile.mkdtemp(prefix=f"sparse_device_explore_extremes_{stamp}_", dir=OUTPUTS))
            save_json(out / "manifest.json", {"identity": identity,
                      "timeout_seconds": args.timeout, "config_count": len(configs)})
        print("Outputs:", out, flush=True)
        interrupted = False

        def stop(_signum, _frame):
            raise KeyboardInterrupt

        previous = signal.signal(signal.SIGTERM, stop)
        try:
            for i, case in enumerate(configs, start=1):
                order = DEVICES if i % 2 else DEVICES[::-1]
                times = []
                for device in order:
                    progress(i - 1, len(configs), f"{case.name} {case.strategy} {device}")
                    row = execute(args, case, device, out, env)
                    times.append(f"{device}: {row['status']} {row['wall_seconds']:.2f}s")
                progress(i, len(configs), f"{case.name} {' | '.join(times)}")
        except KeyboardInterrupt:
            interrupted = True
        finally:
            signal.signal(signal.SIGTERM, previous)
            passed = summarize(out, configs)
            print("Summary:", out / "summary.md")
        return 130 if interrupted else int(passed != 2 * len(configs))


if __name__ == "__main__":
    sys.exit(main())
