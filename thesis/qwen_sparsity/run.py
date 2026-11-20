"""Offline Qwen BF16/FA3 activation profiling with atomic per-generation resume."""

import argparse
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import csv
import fcntl
import functools
import gc
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

from metrics import Collector, scan_weights
from checkpoint import discover_checkpoint, verify_checkpoint
from prepare import prepare_prompts
from protocol import (BLOCKS, DECODE_STEPS, KINDS, LENGTHS, MAG_EDGES, MAX_NEW_TOKENS,
                      METRICS, MODEL, PREFILL_ROWS, PROMPTS, THRESHOLDS, archive_attempt,
                      atomic_json, complete_result, digest, make_cases, read_tokens, sha, stamp)

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("THESIS_QWEN_REPO", str(HERE.parents[1]))).resolve()
OUTPUTS = REPO / "thesis/outputs"


class Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log

    def write(self, value):
        self.original.write(value)
        self.log.write(value)
        self.log.flush()
        return len(value)

    def flush(self):
        self.original.flush()
        self.log.flush()

    def isatty(self):
        return False


def progress(done, total, message):
    width = 24
    fill = width * done // max(1, total)
    print(f"[{'#' * fill}{'-' * (width - fill)}] {done}/{total} {message}", flush=True)


def code_identity():
    return {p: sha(HERE / p) for p in ("protocol.py", "prepare.py", "metrics.py", "checkpoint.py", "run.py")}


def baseline(explicit):
    candidates = [Path(explicit)] if explicit else sorted(OUTPUTS.glob("qwen_baseline_*"), reverse=True)
    for folder in candidates:
        try:
            info = json.loads((folder / "model-info.json").read_text())
            config = json.loads((folder / "model-config.json").read_text())
            result = json.loads((folder / "result.json").read_text())
            rope = config.get("rope_scaling") or {}
            if (info.get("model") == MODEL and info.get("dtype") == "bfloat16"
                    and info.get("attention") == "flash_attention_3" and info.get("B") == 1
                    and result.get("status") == "success"
                    and result.get("cache_length") == info.get("T")
                    and re.fullmatch(r"[0-9a-f]{40}", info.get("revision", ""))
                    and config.get("model_type") == "qwen2"
                    and config.get("max_position_embeddings") == 131072
                    and rope.get("rope_type", rope.get("type")) == "yarn"
                    and rope.get("factor") == 4.0):
                return folder, info, config
        except (OSError, ValueError, TypeError):
            continue
    raise RuntimeError("No successful local Qwen32B BF16/FA3 YaRN4 baseline found. "
                       "Pass --baseline-dir pointing to its existing output folder.")


def model_setup(args):
    if args.model_dir is None:
        try:
            folder, info, config = baseline(args.baseline_dir)
            return str(folder), info, config
        except RuntimeError:
            if args.baseline_dir:
                raise
    source, config = discover_checkpoint(args.cache_dir, args.model_dir)
    print(f"Using existing local Qwen checkpoint: {source['folder']}", flush=True)
    print("No matching historical baseline required. Runtime config: YaRN factor 4, max context 131072; source config.json is unchanged.", flush=True)
    return None, {"revision": source["revision"], "model_source": source["folder"], "checkpoint": source}, config


def environment():
    from importlib.metadata import version
    import torch
    import numpy
    import matplotlib
    actual = {"torch": torch.__version__, "transformers": version("transformers"),
              "flash_attn_3": version("flash-attn-3")}
    expected = {"torch": "2.11.0+cu128", "transformers": "4.57.1", "flash_attn_3": "3.0.1"}
    if actual != expected:
        raise RuntimeError(f"Use the original fa3 environment. Expected {expected}; found {actual}. No installs attempted.")
    if os.environ.get("THESIS_HOST_MALLOC", "0") not in ("", "0"):
        raise RuntimeError("Run with the original CUDA allocator; THESIS_HOST_MALLOC is enabled.")
    if not torch.cuda.is_available() or "GH200" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This profiling protocol requires GH200.")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    actual.update(cuda=torch.version.cuda, gpu=properties.name,
                  gpu_uuid=str(getattr(properties, "uuid", "")),
                  total_memory=properties.total_memory,
                  numpy=numpy.__version__, matplotlib=matplotlib.__version__)
    # Respect other users; this never terminates a process it did not start.
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True, timeout=15)
    foreign = [p.strip() for p in apps.splitlines() if p.strip().isdigit() and int(p) != os.getpid()]
    if foreign:
        raise RuntimeError("GPU has other compute clients: " + ", ".join(foreign))
    return actual


def load_model(manifest, out):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    config_dict = dict(manifest["config"])
    kind = config_dict.pop("model_type")
    config = AutoConfig.for_model(kind, **config_dict)
    started = time.perf_counter()
    print("Loading the existing local Qwen checkpoint once; no model download/build", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        manifest.get("model_source", MODEL), revision=manifest["revision"], cache_dir=manifest["cache_dir"],
        config=config, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="flash_attention_3", use_safetensors=True)
    model.eval()
    if ({str(p.device) for p in model.parameters()} != {"cuda:0"}
            or any(p.dtype != torch.bfloat16 for p in model.parameters() if p.is_floating_point())
            or getattr(model, "is_quantized", False)
            or model.config._attn_implementation != "flash_attention_3"):
        raise RuntimeError("Expected unquantized BF16 Qwen with all weights on cuda:0 and FA3.")
    torch.cuda.synchronize()
    atomic_json(out / ("load_" + str(time.time_ns()) + ".json"),
                {"seconds": time.perf_counter() - started, "at": stamp(),
                 "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters())})
    return model


def verify_hooks(model, collector, tokenizer):
    """Short read-only equivalence check; not one of the 2,100 generations."""
    import torch
    import flash_attn_interface
    from transformers import modeling_flash_attention_utils as futils
    futils.lazy_import_flash_attention("flash_attention_3", force_import=True)
    originals = {"regular": futils._flash_fn, "varlen": futils._flash_varlen_fn}
    if (originals["regular"] is not flash_attn_interface.flash_attn_func
            or originals["varlen"] is not flash_attn_interface.flash_attn_varlen_func):
        raise RuntimeError("Transformers resolved a different FlashAttention implementation.")
    counts = {key: 0 for key in originals}
    def counted(fn, key):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            counts[key] += 1
            return fn(*args, **kwargs)
        return wrapped
    futils._flash_fn = counted(originals["regular"], "regular")
    futils._flash_varlen_fn = counted(originals["varlen"], "varlen")
    ids = tokenizer("Explain why the sky looks blue.", return_tensors="pt")["input_ids"].to("cuda:0")
    try:
        with torch.inference_mode():
            collector.phase = None
            before = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits.clone()
            collector.phase, collector.input_tokens, collector.step = "prefill", ids.shape[1], 0
            after = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            if not torch.isfinite(after).all().item():
                raise RuntimeError("Non-finite logits in instrumentation check.")
            if sum(counts.values()) != 2 * len(model.model.layers):
                raise RuntimeError("Short validation did not invoke FA3 once per layer per forward.")
        return {"status": "pass", "unprofiled_vs_profiled_logits": "bitwise equal",
                "fa3_calls": counts, "forward_calls": 2}
    finally:
        futils._flash_fn, futils._flash_varlen_fn = originals["regular"], originals["varlen"]
        collector.clear()


def execute(model, tokenizer, collector, case, prompt, manifest, folder):
    import numpy as np
    import torch
    started = time.perf_counter()
    protocol = manifest["protocol"]
    framing = read_tokens(Path(manifest["out"]) / "prompts" / prompt["tokens_file"])
    from prepare import context_slice
    budget = case["input_tokens"] - len(framing["prefix"]) - len(framing["suffix"])
    input_ids = framing["prefix"] + context_slice(framing["context"], budget) + framing["suffix"]
    if len(input_ids) != case["input_tokens"]:
        raise RuntimeError("Input length differs from requested configuration.")
    generated, timings, observed_steps = [], [], []
    collector.clear()
    collector.input_tokens = case["input_tokens"]
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    ids = torch.tensor([input_ids], device="cuda:0", dtype=torch.long)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    cache = None
    checks = []
    with (folder / "steps.jsonl").open("w") as log, torch.inference_mode():
        for step in range(protocol["max_new_tokens"]):
            collector.phase = "prefill" if step == 0 else "decode"
            collector.step = step
            if step in collector.decode_steps:
                observed_steps.append(step)
            atomic_json(folder / "progress.json", {"stage": collector.phase, "decode_step": step,
                                                    "elapsed_seconds": time.perf_counter() - started})
            t0 = time.perf_counter()
            outputs = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1,
                            output_attentions=False, output_hidden_states=False)
            next_id = outputs.logits[:, -1].argmax(-1, keepdim=True)
            checks.append(torch.isfinite(outputs.logits).all())
            cache = outputs.past_key_values
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            token = int(next_id.item())
            generated.append(token)
            timing = {"phase": collector.phase, "decode_step": step, "seconds": elapsed,
                      "profiled": step == 0 or step in collector.decode_steps}
            timings.append(timing)
            log.write(json.dumps(timing) + "\n")
            log.flush()
            if token in eos:
                break
            ids = next_id
        if not torch.stack(checks).all().item():
            raise RuntimeError("Non-finite generation logits.")
        if int(cache.get_seq_length()) != case["input_tokens"] + len(generated) - 1:
            raise RuntimeError("Unexpected KV cache length.")
    peak = {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())}
    collector.phase = None
    observations = collector.save(folder / "observations.npz", len(model.model.layers), observed_steps)
    atomic_json(folder / "generation.json", {"token_ids": generated,
                "text": tokenizer.decode(generated, skip_special_tokens=True),
                "stopped_on_eos": bool(generated and generated[-1] in eos),
                "input_token_ids_sha256": digest(input_ids),
                "sampled_decode_steps": observed_steps})
    npz = np.load(folder / "observations.npz", allow_pickle=False)
    widths = npz["metadata"][:, 5].astype(np.float64)
    coverage = {"sampled_elements": int(widths.sum()), "observations": observations}
    npz.close()
    row = {**case, **peak, **coverage, "identity": manifest["identity"], "status": "pass",
           "generated_tokens": len(generated), "finished_at": stamp(),
           "prefill_profiled_seconds": timings[0]["seconds"],
           "decode_seconds": sum(x["seconds"] for x in timings[1:]),
           "wall_seconds": time.perf_counter() - started,
           "timing_note": "Instrumented diagnostic; not baseline inference latency",
           "artifacts": {name: sha(folder / name) for name in
                         ("observations.npz", "generation.json", "steps.jsonl")}}
    # This is the commit marker. A case without this valid marker is retried.
    atomic_json(folder / "result.json", row)
    collector.clear()
    return row


def summarize(out, manifest, known=None):
    rows = list(known.values()) if known is not None else []
    if known is None:
        for case in manifest["cases"]:
            path = out / "cases" / case["id"] / "result.json"
            if path.exists():
                row = json.loads(path.read_text())
                if row.get("identity") != manifest["identity"]:
                    raise RuntimeError(f"Result identity mismatch: {path}")
                rows.append(row)
    fields = ["id", "prompt_index", "source_id", "domain", "input_tokens", "status", "generated_tokens",
              "observations", "prefill_profiled_seconds", "decode_seconds", "wall_seconds",
              "peak_allocated_bytes", "peak_reserved_bytes", "error"]
    tmp = out / "runs.csv.tmp"
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(out / "runs.csv")
    from collections import Counter
    counts = Counter(row["status"] for row in rows)
    pending = len(manifest["cases"]) - counts["pass"] - counts["oom"]
    atomic_json(out / "summary.json", {"expected_runs": len(manifest["cases"]), "statuses": dict(counts),
                "all_pass": counts["pass"] == len(manifest["cases"]),
                "pending_or_retryable": pending, "unattempted": len(manifest["cases"]) - len(rows), "updated_at": stamp()})
    (out / "summary.md").write_text(
        "# Qwen sparsity profiling\n\n"
        f"- Expected generations: {len(manifest['cases'])}\n"
        f"- Recorded statuses: {dict(counts)}\n"
        f"- Pending or interrupted/error retries: {pending}\n\n"
        "Exact-zero sparsity and near-zero thresholds are separate. Hooks sample token rows; "
        "they do not scan all prefill activations. Statistics describe the selected long-source "
        "LongBench-v2 subset. Timings include instrumentation.\n", encoding="utf-8")
    return counts


def stop_signal(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--baseline-dir")
    parser.add_argument("--model-dir", type=Path, help="Existing local checkpoint directory or Hugging Face model cache; no baseline artifacts required")
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / "models/huggingface")
    parser.add_argument("--dataset-cache", type=Path, default=Path.home() / ".cache/thesis_qwen_sparsity")
    parser.add_argument("--smoke", action="store_true", help="4 short generations, separate output directory")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--max-runs", type=int, help="Stop cleanly after N new successful/attempted runs this invocation")
    parser.add_argument("--plot-only", type=Path, help="Rebuild plots from saved runs without model/GPU")
    args = parser.parse_args()
    if args.plot_only:
        from plot import plot
        plot(args.plot_only.resolve())
        return 0
    if args.max_runs is not None and args.max_runs < 1:
        parser.error("--max-runs must be positive")
    if args.resume and (args.smoke or args.baseline_dir or args.model_dir):
        parser.error("Resume uses the saved protocol; do not combine --resume with --smoke/--baseline-dir/--model-dir")
    if args.baseline_dir and args.model_dir:
        parser.error("Choose --baseline-dir or --model-dir, not both")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[name] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    OUTPUTS.mkdir(exist_ok=True)
    with ExitStack() as stack:
        for name in (".tma_sweep.lock", ".qwen-stress.lock", ".qwen-sparsity.lock"):
            lock = stack.enter_context((OUTPUTS / name).open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Another thesis runner holds " + name + "; wait for it to finish.")
        env = environment()
        from transformers import AutoTokenizer
        if args.resume:
            out = args.resume.resolve()
            ready = (out / "manifest.json").exists()
            manifest = json.loads((out / ("manifest.json" if ready else "preparing.json")).read_text())
            if ready and digest({k: v for k, v in manifest.items() if k not in ("created_at", "out", "identity")}) != manifest["identity"]:
                raise RuntimeError("Saved manifest changed.")
            if manifest["code"] != code_identity() or manifest["environment"] != env:
                raise RuntimeError("Code/environment changed. Resume with the original code and fa3 environment.")
            if str(out) != manifest["out"]:
                raise RuntimeError("Resume from the original output directory.")
            for p in manifest.get("prompts", []):
                if sha(out / "prompts" / p["tokens_file"]) != p["tokens_sha256"]:
                    raise RuntimeError("Saved prompt tokens changed.")
            if manifest.get("checkpoint"):
                verify_checkpoint(manifest["checkpoint"])
        else:
            ready = False
            folder, info, config = model_setup(args)
            mode = "smoke" if args.smoke else "full"
            out = Path(tempfile.mkdtemp(prefix=f"qwen_sparsity_{mode}_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_", dir=OUTPUTS))
            manifest = {"out": str(out), "baseline": folder, "revision": info["revision"], "config": config,
                        "model_source": info.get("model_source", MODEL), "checkpoint": info.get("checkpoint"),
                        "cache_dir": str(args.cache_dir.resolve()), "dataset_cache": str(args.dataset_cache.resolve()),
                        "environment": env, "code": code_identity(),
                        "created_at": stamp(), "protocol": {
                            "lengths": [1024, 2048] if args.smoke else list(LENGTHS),
                            "prompts": 2 if args.smoke else PROMPTS,
                            "max_new_tokens": 4 if args.smoke else MAX_NEW_TOKENS,
                            "prefill_rows": 4 if args.smoke else PREFILL_ROWS,
                            "decode_steps": [1, 2, 3] if args.smoke else list(DECODE_STEPS),
                            "scan_weights": not args.smoke, "batch": 1, "seed": 711,
                            "greedy": True, "stop_on_eos": True, "kinds": KINDS,
                            "thresholds": THRESHOLDS, "block_widths": BLOCKS,
                            "magnitude_edges": MAG_EDGES, "metrics": METRICS,
                            "block_definition": "contiguous nonoverlapping feature-axis blocks in one token row; full blocks only"}}
            snapshot = out / "runner_source"
            snapshot.mkdir()
            for file in HERE.glob("*.py"):
                shutil.copy2(file, snapshot / file.name)
            atomic_json(out / "preparing.json", manifest)
        log = stack.enter_context((out / "run.log").open("a", encoding="utf-8", buffering=1))
        stack.enter_context(redirect_stdout(Tee(sys.stdout, log)))
        stack.enter_context(redirect_stderr(Tee(sys.stderr, log)))
        print(f"Outputs: {out}", flush=True)
        print(f"Resume command: bash thesis/scripts/run_qwen_sparsity.sh --resume {out}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(manifest.get("model_source", MODEL), revision=manifest["revision"],
                       cache_dir=manifest["cache_dir"], local_files_only=True, trust_remote_code=False)
        if not ready:
            p = manifest["protocol"]
            manifest["prompts"] = prepare_prompts(out, tokenizer, p["prompts"], p["lengths"], manifest["dataset_cache"])
            manifest["cases"] = make_cases(manifest["prompts"], p["lengths"])
            manifest["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
            manifest["identity"] = digest({k: v for k, v in manifest.items() if k not in ("created_at", "out")})
            atomic_json(out / "manifest.json", manifest)
            (out / "preparing.json").unlink()
        (out / "cases").mkdir(exist_ok=True)
        done, pending, known = 0, [], {}
        for case in manifest["cases"]:
            folder = out / "cases" / case["id"]
            folder.mkdir(exist_ok=True)
            complete = complete_result(folder, manifest["identity"], case)
            prior = folder / "result.json"
            if prior.exists():
                known[case["id"]] = json.loads(prior.read_text())
            if complete:
                done += 1
            elif prior.exists() and json.loads(prior.read_text()).get("status") == "oom" and not args.retry_failed:
                done += 1
            else:
                pending.append(case)
        total = len(manifest["cases"])
        print(f"Plan: {manifest['protocol']['prompts']} prompts x {len(manifest['protocol']['lengths'])} lengths = {total} generations", flush=True)
        print(f"Saved: {done}; pending/retries: {len(pending)}", flush=True)
        if not pending:
            counts = summarize(out, manifest, known)
            from plot import plot
            plot(out)
            return 2 if counts["oom"] else 0
        import torch
        model, collector = None, None
        return_code = 0
        for signum in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, stop_signal)
        try:
            model = load_model(manifest, out)
            torch.manual_seed(manifest["protocol"]["seed"])
            p = manifest["protocol"]
            collector = Collector(model, p["prefill_rows"], p["decode_steps"])
            atomic_json(out / ("validation_" + str(time.time_ns()) + ".json"), verify_hooks(model, collector, tokenizer))
            gc.collect()
            torch.cuda.empty_cache()
            if p["scan_weights"] and not (out / "weights.json").exists():
                atomic_json(out / "weights.json", {"identity": manifest["identity"], "parameters": scan_weights(model, progress)})
            elif (out / "weights.json").exists():
                if json.loads((out / "weights.json").read_text())["identity"] != manifest["identity"]:
                    raise RuntimeError("Weight scan identity mismatch.")
            attempted = 0
            for case in pending:
                folder = out / "cases" / case["id"]
                archive_attempt(folder)
                progress(done, total, case["id"] + " start")
                started = time.perf_counter()
                try:
                    row = execute(model, tokenizer, collector, case, manifest["prompts"][case["prompt_index"]], manifest, folder)
                except BaseException as exc:
                    status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "error"
                    row = {**case, "identity": manifest["identity"], "status": status,
                           "wall_seconds": time.perf_counter() - started, "error": str(exc), "finished_at": stamp()}
                    atomic_json(folder / "result.json", row)
                    known[case["id"]] = row
                    if status != "oom":
                        raise
                    print(f"{case['id']}: OOM recorded; configuration was not reduced", flush=True)
                finally:
                    collector.clear()
                    gc.collect()
                    torch.cuda.empty_cache()
                done += 1
                attempted += 1
                known[case["id"]] = row
                progress(done, total, f"{case['id']} {row['status']} {row['wall_seconds']:.2f}s")
                summarize(out, manifest, known)
                if args.max_runs and attempted >= args.max_runs:
                    break
        except KeyboardInterrupt:
            print("Stopped. Completed generations are saved; the interrupted one will restart on resume.", flush=True)
            return_code = 130
        except Exception:
            traceback.print_exc()
            return_code = 1
        finally:
            if collector is not None:
                collector.close()
            del model
            gc.collect()
            torch.cuda.empty_cache()
            counts = summarize(out, manifest, known)
            print(f"Resume: bash thesis/scripts/run_qwen_sparsity.sh --resume {out}", flush=True)
        # Plotting is CPU-only and happens after releasing model tensors.
        if return_code != 1:
            try:
                from plot import plot
                plot(out)
            except Exception:
                traceback.print_exc()
                print("Saved results are intact. Retry plotting with --plot-only <output directory>.", flush=True)
        return 2 if return_code == 0 and counts["oom"] else return_code


if __name__ == "__main__":
    sys.exit(main())
