#!/usr/bin/env bash
# Qwen32B full-model prefill capacity sweep. Uses the completed local baseline.
# No installs, builds, downloads, CPU weight offload or checkpoint writes.
# Usage: bash thesis/scripts/run_qwen_stress.sh [--max-t 1048576] [--no-refine]
# Resume: bash thesis/scripts/run_qwen_stress.sh --resume /absolute/run/directory
set -Eeuo pipefail
umask 077
script_path="$(realpath "${BASH_SOURCE[0]}")"
repo_root="$(git -C "$(dirname "$script_path")" rev-parse --show-toplevel)"
cd "$repo_root"
[[ "$(git branch --show-current)" == main ]] || { echo 'Run on main; no branch was changed.' >&2; exit 2; }
mkdir -p "$repo_root/thesis/outputs"
exec 9>"$repo_root/thesis/outputs/.qwen-stress.lock"
flock -n 9 || { echo 'Another Qwen stress runner holds this repository lock.' >&2; exit 2; }
resuming=0
if [[ "${1:-}" == --resume ]]; then
    [[ $# -ge 2 ]] || { echo '--resume needs a run directory.' >&2; exit 2; }
    run_dir="$(realpath "$2")"
    [[ "$run_dir" == "$repo_root"/thesis/outputs/qwen_stress_* ]] || { echo 'Resume directory must be a qwen_stress run inside thesis/outputs.' >&2; exit 2; }
    cmp -s "$script_path" "$run_dir/runner-used.sh" || { echo 'Resume with the original runner-used.sh copied back to thesis/scripts; protocol must stay identical.' >&2; exit 2; }
    shift 2
    resuming=1
else
    run_dir="$(mktemp -d "$repo_root/thesis/outputs/qwen_stress_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
    cp "$script_path" "$run_dir/runner-used.sh"
fi
exec > >(tee -a "$run_dir/run.log") 2>&1
finish() {
    result_code=$?
    trap - EXIT
    printf 'exit_code=%s\n' "$result_code" > "$run_dir/status.txt"
    echo "Results: $run_dir/results.csv"
    echo "Summary: $run_dir/summary.json"
    echo "Exit code: $result_code"
    exit "$result_code"
}
trap finish EXIT
echo "Run directory: $run_dir"
set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate fa3
set -u
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
git rev-parse HEAD > "$run_dir/source-commit-current.txt"
git status --short > "$run_dir/source-status-current.txt"
cat > "$run_dir/stress_driver.py" <<'QWEN_STRESS_PY'
"""Fixed-protocol, offline Qwen capacity experiment. stdlib-only controller."""
import argparse
import csv
import functools
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone

MODEL = "Qwen/Qwen2.5-32B-Instruct"
GIB = 2 ** 30
MEMORY_STAGES = {"input_allocation", "prefill"}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def find_baseline(repo, explicit=None):
    candidates = [Path(explicit)] if explicit else sorted(
        (Path(repo) / "thesis/outputs").glob("qwen_baseline_*"), reverse=True)
    for path in candidates:
        try:
            info = json.loads((path / "model-info.json").read_text())
            result = json.loads((path / "result.json").read_text())
            config = json.loads((path / "model-config.json").read_text())
            valid = (info.get("model") == MODEL and info.get("B") == 1
                     and info.get("dtype") == "bfloat16"
                     and info.get("attention") == "flash_attention_3"
                     and result.get("status") == "success"
                     and result.get("cache_length") == info.get("T")
                     and re.fullmatch(r"[0-9a-f]{40}", info.get("revision", "")))
            rope = config.get("rope_scaling") or {}
            valid = valid and (config.get("model_type") == "qwen2"
                and config.get("max_position_embeddings") == 131072
                and rope.get("rope_type", rope.get("type")) == "yarn"
                and rope.get("factor") == 4.0
                and rope.get("original_max_position_embeddings") == 32768)
            if valid:
                return path, info, config
        except (OSError, ValueError, TypeError):
            pass
    raise RuntimeError("No successful Qwen32B BF16/FA3 baseline with the existing YaRN4/128k config. Supply --baseline-dir; do not rebuild/redownload.")


def coarse_lengths(max_t):
    lengths, t = [], 1024
    while t <= max_t:
        lengths.append(t)
        t *= 2
    if lengths[-1] != max_t:
        lengths.append(max_t)
    return lengths


def refine_midpoint(low, high, step=1024):
    if high - low <= step:
        return None
    return ((low + high) // (2 * step)) * step


def is_capacity_oom(row):
    return row.get("status") == "cuda_oom" and row.get("stage") in MEMORY_STAGES


def stop_group(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=8)
    except ProcessLookupError:
        pass


def append_result(run_dir, row):
    # JSONL is authoritative. Rebuild CSV after every complete JSONL append.
    path = Path(run_dir) / "results.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    rebuild_csv(run_dir)


CSV_FIELDS = ["T", "B", "status", "stage", "context_class", "prefill_seconds",
    "baseline_allocated_bytes", "before_forward_allocated_bytes", "peak_allocated_bytes",
    "peak_allocated_gib", "peak_reserved_bytes", "peak_reserved_gib", "allocated_after_bytes",
    "peak_minus_baseline_bytes", "device_total_bytes", "device_used_before_bytes",
    "device_used_after_bytes", "gpu_used_sampled_peak_mib", "cpu_process_peak_rss_bytes",
    "forward_sampled_peak_rss_bytes", "cache_length", "fa3_calls", "expected_kv_bytes",
    "exit_code", "trial_dir", "error"]


def read_results(run_dir):
    path = Path(run_dir) / "results.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            # A torn trailing append is recovered separately; never treat it as a result.
            if not line.endswith("\n"):
                raise RuntimeError("Truncated results.jsonl tail; preserve it and recover the complete per-trial result.json before resuming.")
            rows.append(json.loads(line))
    return rows


def recover_index_tail(run_dir):
    """Preserve a torn final append; its complete worker artifact can be re-indexed."""
    path = Path(run_dir) / "results.jsonl"
    if not path.exists():
        return None
    raw = path.read_bytes()
    if not raw or raw.endswith(b"\n"):
        return None
    end = raw.rfind(b"\n") + 1
    saved = path.with_name("results-torn-tail-" + str(time.time_ns()) + ".txt")
    saved.write_bytes(raw[end:])
    temp = path.with_suffix(".jsonl.recovered")
    with temp.open("wb") as f:
        f.write(raw[:end])
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    return str(saved)


def rebuild_csv(run_dir):
    path = Path(run_dir) / "results.csv"
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(read_results(run_dir))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def worker(manifest_path, t, trial_dir):
    # Imports stay in the worker: the controller never holds a CUDA context/model.
    row = {"T": t, "B": 1, "status": "error", "stage": "imports", "started_at": stamp(),
           "context_class": "within_128k" if t <= 131072 else "beyond_128k_capacity_only"}
    manifest = json.loads(Path(manifest_path).read_text())
    row["protocol_sha256"] = manifest["protocol_sha256"]
    torch = None
    peak_ready = False
    sampler_stop = threading.Event()
    sampler = None
    counters = {"regular": 0, "varlen": 0}
    rss_samples = []
    started = time.monotonic()
    try:
        from importlib.metadata import version
        import torch as torch_module
        torch = torch_module
        from transformers import AutoConfig, AutoModelForCausalLM
        import flash_attn_interface
        from transformers import modeling_flash_attention_utils as futils
        manifest = json.loads(Path(manifest_path).read_text())
        protocol = manifest["protocol"]
        row["protocol_sha256"] = manifest["protocol_sha256"]
        row["versions"] = {"torch": torch.__version__, "transformers": version("transformers"),
                           "flash_attn_3": version("flash-attn-3")}
        if row["versions"] != {"torch": "2.11.0+cu128", "transformers": "4.57.1", "flash_attn_3": "3.0.1"}:
            raise RuntimeError("Environment differs from the completed baseline; no automatic installs are permitted.")
        if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (9, 0):
            raise RuntimeError("Requires the allocated GH200/Hopper GPU, capability9.0.")
        torch.cuda.set_device(0)
        torch.cuda.init()
        props = torch.cuda.get_device_properties(0)
        row.update(gpu_name=props.name, gpu_uuid=str(getattr(props, "uuid", "")),
                   cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), device_total_bytes=int(props.total_memory))
        # Other compute clients on the target GPU invalidate a capacity measurement.
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
                              text=True, capture_output=True, timeout=10)
        if apps.returncode:
            raise RuntimeError("Cannot verify GPU exclusivity with nvidia-smi: " + apps.stderr.strip())
        clients = list(csv.reader(apps.stdout.splitlines()))
        own = [r[1].strip() for r in clients if len(r) == 2 and r[0].strip() == str(os.getpid())]
        uuid = own[0] if own else row["gpu_uuid"]
        if uuid and re.fullmatch(r"(?:GPU-)?[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", uuid):
            if not uuid.startswith("GPU-"):
                uuid = "GPU-" + uuid
        else:
            uuid = None  # Unknown mapping: conservatively reject any foreign compute client.
        foreign = [r for r in clients if len(r) == 2 and r[0].strip().isdigit()
                   and r[0].strip() != str(os.getpid()) and (not uuid or r[1].strip() == uuid)]
        if foreign:
            row["status"] = "gpu_busy"
            raise RuntimeError("Another compute process occupies the target GPU; capacity run stopped.")
        row["gpu_uuid"] = uuid
        row["stage"] = "model_load"
        config_dict = json.loads((Path(manifest_path).parent / "baseline-model-config.json").read_text())
        model_type = config_dict.pop("model_type")
        config = AutoConfig.for_model(model_type, **config_dict)
        model = AutoModelForCausalLM.from_pretrained(MODEL, revision=protocol["revision"],
            cache_dir=protocol["cache_dir"], config=config, local_files_only=True,
            dtype=torch.bfloat16, device_map={"": "cuda:0"},
            attn_implementation="flash_attention_3", use_safetensors=True)
        model.eval()
        if {str(p.device) for p in model.parameters()} != {"cuda:0"}:
            raise RuntimeError("Some model weights are outside cuda:0.")
        if getattr(model, "is_quantized", False):
            raise RuntimeError("Quantized model is outside this BF16 protocol.")
        if any(p.dtype != torch.bfloat16 for p in model.parameters() if p.is_floating_point()):
            raise RuntimeError("Some floating model parameters are not BF16.")
        if model.config._attn_implementation != "flash_attention_3":
            raise RuntimeError("Incorrect attention backend.")
        row["flash_attn_interface_file"] = str(Path(flash_attn_interface.__file__).resolve())
        row["parameter_devices"] = sorted({str(p.device) for p in model.parameters()})
        row["hf_device_map"] = {k: str(v) for k, v in getattr(model, "hf_device_map", {}).items()}
        row["parameter_count"] = sum(p.numel() for p in model.parameters())
        row["parameter_bytes"] = sum(p.numel() * p.element_size() for p in model.parameters())
        # Verify actual installed FA3 entry calls, preserving original function signatures.
        futils.lazy_import_flash_attention("flash_attention_3", force_import=True)
        if futils._flash_fn is not flash_attn_interface.flash_attn_func or futils._flash_varlen_fn is not flash_attn_interface.flash_attn_varlen_func:
            raise RuntimeError("HF resolved a different FlashAttention implementation.")
        def counted(fn, key):
            @functools.wraps(fn)
            def wrapped(*args, **kwargs):
                counters[key] += 1
                return fn(*args, **kwargs)
            return wrapped
        futils._flash_fn = counted(futils._flash_fn, "regular")
        futils._flash_varlen_fn = counted(futils._flash_varlen_fn, "varlen")
        row["stage"] = "warmup"
        torch.manual_seed(42)
        warm_ids = torch.randint(0, config.vocab_size, (1, 16), device="cuda:0", dtype=torch.long)
        with torch.inference_mode():
            warm = model(input_ids=warm_ids, use_cache=True, logits_to_keep=1,
                         output_attentions=False, output_hidden_states=False)
        torch.cuda.synchronize()
        if sum(counters.values()) != config.num_hidden_layers:
            raise RuntimeError("Warmup did not call FA3 once for every transformer layer.")
        del warm, warm_ids
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        row["baseline_allocated_bytes"] = int(torch.cuda.memory_allocated(0))
        row["baseline_reserved_bytes"] = int(torch.cuda.memory_reserved(0))
        free, total = torch.cuda.mem_get_info(0)
        row.update(device_free_before_bytes=int(free), device_total_bytes=int(total),
                   device_used_before_bytes=int(total - free))
        # Reset BEFORE the input allocation so its memory is included in the trial peak.
        torch.cuda.reset_peak_memory_stats(0)
        peak_ready = True
        counters.update(regular=0, varlen=0)
        row["expected_kv_bytes"] = int(2 * config.num_hidden_layers * t * config.num_key_value_heads
            * (config.hidden_size // config.num_attention_heads) * 2)
        row["stage"] = "input_allocation"
        torch.manual_seed(42)
        input_ids = torch.randint(0, config.vocab_size, (1, t), device="cuda:0", dtype=torch.long)
        torch.cuda.synchronize()
        row["before_forward_allocated_bytes"] = int(torch.cuda.memory_allocated(0))
        def sample_rss():
            while not sampler_stop.is_set():
                try:
                    for line in Path("/proc/self/status").read_text().splitlines():
                        if line.startswith("VmRSS:"):
                            rss_samples.append(int(line.split()[1]) * 1024)
                            break
                except OSError:
                    pass
                sampler_stop.wait(0.05)
        sampler = threading.Thread(target=sample_rss, daemon=True)
        sampler.start()
        row["stage"] = "prefill"
        write_json(Path(trial_dir) / "progress.json", row)
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, use_cache=True, logits_to_keep=1,
                            output_attentions=False, output_hidden_states=False)
        torch.cuda.synchronize()
        row["prefill_seconds"] = time.perf_counter() - start
        # Snapshot BEFORE finite/logit validation and before releasing KV/output tensors.
        row["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(0))
        row["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(0))
        row["allocated_after_bytes"] = int(torch.cuda.memory_allocated(0))
        free, total = torch.cuda.mem_get_info(0)
        row["device_used_after_bytes"] = int(total - free)
        peak_ready = False  # Preserve the captured measurement during later validation.
        row["stage"] = "validation"
        row["cache_length"] = int(outputs.past_key_values.get_seq_length())
        row["logits_shape"] = list(outputs.logits.shape)
        row["fa3_calls"] = sum(counters.values())
        if row["cache_length"] != t or row["logits_shape"] != [1, 1, config.vocab_size]:
            raise RuntimeError("Unexpected full-prefill output/cache dimensions.")
        if row["fa3_calls"] != config.num_hidden_layers:
            raise RuntimeError("Measured prefill did not call FA3 once per layer.")
        if not torch.isfinite(outputs.logits).all().item():
            raise RuntimeError("Non-finite final-token logits.")
        row.update(status="success", stage="complete")
    except BaseException as exc:
        if torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError):
            row["status"] = "cuda_oom"
        elif isinstance(exc, KeyboardInterrupt):
            row["status"] = "interrupted"
        row["error"] = str(exc)
        row["exception_type"] = type(exc).__name__
        traceback.print_exc()
    finally:
        sampler_stop.set()
        if sampler is not None:
            sampler.join(timeout=1)
        if torch is not None and peak_ready:
            for key, fn in [("peak_allocated_bytes", torch.cuda.max_memory_allocated),
                            ("peak_reserved_bytes", torch.cuda.max_memory_reserved)]:
                try:
                    row[key] = int(fn(0))
                except Exception:
                    pass
        for prefix in ["peak_allocated", "peak_reserved"]:
            if prefix + "_bytes" in row:
                row[prefix + "_gib"] = row[prefix + "_bytes"] / GIB
        if "peak_allocated_bytes" in row and "baseline_allocated_bytes" in row:
            row["peak_minus_baseline_bytes"] = row["peak_allocated_bytes"] - row["baseline_allocated_bytes"]
        row["fa3_calls"] = sum(counters.values())
        row["cpu_process_peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        row["forward_sampled_peak_rss_bytes"] = max(rss_samples) if rss_samples else None
        row["worker_wall_seconds"] = time.monotonic() - started
        row["ended_at"] = stamp()
        write_json(Path(trial_dir) / "result.json", row)
        print(json.dumps(row, indent=2), flush=True)
    return 0 if row["status"] == "success" else (10 if row["status"] == "cuda_oom" else 2)


def sampled_gpu_peak(path, uuid):
    try:
        with Path(path).open() as f:
            values = [float(r[3].strip()) for r in csv.reader(f) if len(r) == 5
                      and uuid and r[2].strip() == uuid and r[3].strip().replace(".", "", 1).isdigit()]
        return max(values) if values else None
    except OSError:
        return None


def run_trial(run_dir, manifest, t, timeout):
    run_dir = Path(run_dir)
    trial = run_dir / f"T_{t:09d}"
    # Completed-but-not-yet-indexed trials can be recovered after a controller interruption.
    if (trial / "result.json").exists():
        prior = json.loads((trial / "result.json").read_text())
        if prior.get("protocol_sha256") == manifest["protocol_sha256"] and prior.get("T") == t:
            if prior.get("status") == "success" or is_capacity_oom(prior):
                prior["recovered_from_worker_artifact"] = True
                prior.setdefault("trial_dir", str(trial))
                return prior
    if trial.exists():
        trial.rename(trial.with_name(trial.name + "_interrupted_" + str(time.time_ns())))
    trial.mkdir()
    write_json(run_dir / "checkpoint.json", {"running_T": t, "trial_dir": str(trial), "updated_at": stamp()})
    monitor = process = None
    status_override = None
    with (trial / "worker.log").open("w") as log, (trial / "gpu-samples.csv").open("w") as gpu, (trial / "gpu-monitor.log").open("w") as gpuerr:
        try:
            try:
                monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,uuid,memory.used,memory.total",
                    "--format=csv,noheader,nounits", "--loop-ms=500"], stdout=gpu, stderr=gpuerr, start_new_session=True)
            except OSError as exc:
                gpuerr.write("GPU sampling unavailable: " + str(exc) + "\n")
            process = subprocess.Popen([sys.executable, "-u", __file__, "--worker", str(run_dir / "manifest.json"),
                str(t), str(trial)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                status_override = "timeout"
            except KeyboardInterrupt:
                status_override = "interrupted"
        finally:
            stop_group(process)
            stop_group(monitor)
    result_path = trial / "result.json"
    row = json.loads(result_path.read_text()) if result_path.exists() else {
        "T": t, "B": 1, "stage": "worker_process", "status": "process_failed",
        "protocol_sha256": manifest["protocol_sha256"],
        "context_class": "within_128k" if t <= 131072 else "beyond_128k_capacity_only",
        "error": "Worker ended without result.json; inspect worker.log. A killed process is not automatically CUDA OOM."}
    if status_override:
        row.update(status=status_override, error="Controller stopped this trial: " + status_override)
    if row.get("status") == "success" and process is not None and process.returncode != 0:
        row.update(status="process_failed", error="Success artifact exists but worker exit was nonzero.")
    row.update(exit_code=process.returncode if process is not None else None, trial_dir=str(trial))
    row["gpu_used_sampled_peak_mib"] = sampled_gpu_peak(trial / "gpu-samples.csv", row.get("gpu_uuid"))
    write_json(result_path, row)
    return row


def sweep(max_t, refine, run_one):
    low, high = None, None
    for t in coarse_lengths(max_t):
        row = run_one(t)
        if row["status"] == "success":
            low = t
        elif is_capacity_oom(row):
            high = t
            break
        else:
            return {"stop_reason": row["status"], "largest_success_T": low, "smallest_capacity_oom_T": high, "last_T": t}
    if refine and low is not None and high is not None:
        while (mid := refine_midpoint(low, high)) is not None:
            row = run_one(mid)
            if row["status"] == "success":
                low = mid
            elif is_capacity_oom(row):
                high = mid
            else:
                return {"stop_reason": row["status"], "largest_success_T": low, "smallest_capacity_oom_T": high, "last_T": mid}
    return {"stop_reason": "capacity_oom_found" if high else "max_t_reached_without_oom",
            "largest_success_T": low, "smallest_capacity_oom_T": high,
            "frontier_resolution_tokens": high - low if high is not None and low is not None else None,
            "frontier_assumption": "monotonic capacity under the same fixed environment; not a proof for all lengths"}


def main():
    parser = argparse.ArgumentParser(description="Offline Qwen32B BF16/FA3 prefill memory sweep; no installations.")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--baseline-dir")
    parser.add_argument("--cache-dir", default=str(Path.home() / "models/huggingface"))
    parser.add_argument("--max-t", type=int, default=1048576)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-wall-seconds", type=int, default=21600)
    parser.add_argument("--no-refine", action="store_true")
    args = parser.parse_args()
    if args.max_t < 1024 or args.max_t > 1048576 or args.max_t % 1024:
        parser.error("--max-t must be a multiple of1024, between1024 and1048576.")
    if args.timeout_seconds <= 0 or args.max_wall_seconds <= 0:
        parser.error("Timeouts must be positive.")
    run_dir = Path(args.run_dir)
    manifest_path = run_dir / "manifest.json"
    if args.resume_existing:
        manifest = json.loads(manifest_path.read_text())
        # Resume always uses the original immutable protocol/cap/time settings.
        if any(x.startswith(("--max-t", "--no-refine", "--baseline-dir", "--cache-dir")) for x in sys.argv[1:]):
            raise RuntimeError("Resume uses saved protocol. Do not pass protocol-changing options.")
        if manifest["driver_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise RuntimeError("Driver changed; resume with the original runner.")
    else:
        baseline, info, config = find_baseline(args.repo_root, args.baseline_dir)
        protocol = {"model": MODEL, "revision": info["revision"], "cache_dir": str(Path(args.cache_dir).resolve()),
            "config_sha256": digest(config), "B": 1, "dtype": "bfloat16", "attention": "flash_attention_3",
            "use_cache": True, "logits_to_keep": 1, "seed": 42, "warmup_T": 16,
            "max_t": args.max_t, "refine": not args.no_refine, "timeout_seconds": args.timeout_seconds,
            "max_wall_seconds": args.max_wall_seconds, "fixed_yarn_factor": 4.0,
            "above_131072": "capacity_only_context_extrapolation_not_quality_validation"}
        manifest = {"schema": 1, "created_at": stamp(), "baseline_dir": str(baseline), "protocol": protocol,
            "protocol_sha256": digest(protocol), "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "memory_note": "Allocator peaks are exact counter readings in bytes; device-wide GPU samples are approximate500ms samples including model load. OOM peaks exclude the rejected allocation. CPU RSS and GPU usage must not be summed as unique physical memory.",
            "timing_note": "One synchronized prefill perT after16-token warmup; not a repeated throughput benchmark."}
        write_json(run_dir / "baseline-model-info.json", info)
        write_json(run_dir / "baseline-model-config.json", config)
        write_json(manifest_path, manifest)
        env = {k: os.environ.get(k) for k in ["CONDA_PREFIX", "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF", "OMP_NUM_THREADS"]}
        write_json(run_dir / "environment.json", {"python": sys.version, "executable": sys.executable, "environment": env})
        for command, filename in [(["nvidia-smi"], "gpu-before.txt"), ([sys.executable, "-m", "pip", "freeze"], "packages.txt")]:
            r = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
            (run_dir / filename).write_text(r.stdout)
    if digest(json.loads((run_dir / "baseline-model-config.json").read_text())) != manifest["protocol"]["config_sha256"]:
        raise RuntimeError("Saved baseline model config changed.")
    protocol = manifest["protocol"]
    if args.resume_existing:
        recovered = recover_index_tail(run_dir)
        if recovered:
            print("Preserved torn JSONL tail at " + recovered, flush=True)
    rows = read_results(run_dir)
    for r in rows:
        if r.get("protocol_sha256") != manifest["protocol_sha256"]:
            raise RuntimeError("Results contain a different protocol.")
    done = {r["T"]: r for r in rows if r["status"] == "success" or is_capacity_oom(r)}
    start = time.monotonic()
    def run_one(t):
        if t in done:
            return done[t]
        remaining = protocol["max_wall_seconds"] - (time.monotonic() - start)
        if remaining <= 0:
            return {"T": t, "status": "wall_time_limit"}
        print(f"\nSTART T={t} B=1 BF16 FA3; context={'within128k' if t <= 131072 else 'EXTRAPOLATIVE_CAPACITY_ONLY'}", flush=True)
        row = run_trial(run_dir, manifest, t, min(protocol["timeout_seconds"], remaining))
        append_result(run_dir, row)
        done[t] = row
        print(f"RESULT T={t} status={row['status']} peak_allocated_GiB={row.get('peak_allocated_gib')} prefill_s={row.get('prefill_seconds')}", flush=True)
        return row
    summary = sweep(protocol["max_t"], protocol["refine"], run_one)
    summary.update(updated_at=stamp(), protocol_sha256=manifest["protocol_sha256"],
        result_rows=len(read_results(run_dir)), configured_model_context_limit=131072,
        model_quality_tested=False, actual_peak_required_for_failed_oom_known=False)
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "checkpoint.json", {"phase": "stopped", **summary})
    print(json.dumps(summary, indent=2), flush=True)
    if summary["stop_reason"] in {"capacity_oom_found", "max_t_reached_without_oom"}:
        print("QWEN_STRESS_COMPLETE", flush=True)
        return 0
    print("QWEN_STRESS_STOPPED_CHECK_SUMMARY_AND_WORKER_LOG", flush=True)
    return 130 if summary["stop_reason"] == "interrupted" else 2


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        # Termination still has an opportunity to save a partial, explicitly interrupted row.
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        sys.exit(worker(sys.argv[2], int(sys.argv[3]), sys.argv[4]))
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted; completed per-T results remain saved. Resume with --resume.", file=sys.stderr)
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(2)
QWEN_STRESS_PY
if [[ "$resuming" == 1 ]]; then
    python -u "$run_dir/stress_driver.py" --repo-root "$repo_root" --run-dir "$run_dir" --resume-existing "$@"
else
    python -u "$run_dir/stress_driver.py" --repo-root "$repo_root" --run-dir "$run_dir" "$@"
fi
