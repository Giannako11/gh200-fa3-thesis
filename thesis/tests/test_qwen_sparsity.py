"""CPU checks for metrics, generation bookkeeping, sampling and resume integrity.

Run with a PyTorch environment: python -m unittest discover -s thesis/tests -p test_qwen_sparsity.py
These tests do not claim GH200/FA3 runtime validation.
"""

import json
import signal
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qwen_sparsity"))
from metrics import Collector, scan_weights, tensor_metrics
from plot import analyze, bootstrap_mean, prompt_distribution
from prepare import context_slice, framing, prepare_prompts
from protocol import (BLOCKS, KINDS, LENGTHS, METRICS, THRESHOLDS, archive_attempt,
                      atomic_json, complete_result, make_cases, positions, read_tokens,
                      sha, stratified_order, write_tokens)
import run as runner
from run import execute, summarize
from checkpoint import EXPECTED, MODEL_CACHE, discover_checkpoint, inspect_checkpoint, runtime_config, verify_checkpoint


class CheckpointTests(unittest.TestCase):
    def checkpoint(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        config = {**EXPECTED, "torch_dtype": "bfloat16", "max_position_embeddings": 32768}
        atomic_json(folder / "config.json", config)
        atomic_json(folder / "model.safetensors.index.json", {"weight_map": {"test": "model-00001.safetensors"}})
        (folder / "model-00001.safetensors").write_bytes(b"unit-test fixture, not model weights")
        for name in ("tokenizer.json", "tokenizer_config.json"):
            atomic_json(folder / name, {})
        return config

    def test_standard_cache_without_baseline_selects_local_ref(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_cache = root / MODEL_CACHE
            folder = model_cache / "snapshots" / ("a"*40)
            original = self.checkpoint(folder)
            (model_cache / "refs").mkdir()
            (model_cache / "refs/main").write_text("a"*40)
            source, config = discover_checkpoint(root)
            self.assertEqual(source["folder"], str(folder))
            self.assertEqual(source["revision"], "a"*40)
            self.assertFalse(source["baseline_success_claimed"])
            self.assertEqual(config["rope_scaling"]["factor"], 4)
            self.assertEqual(config["max_position_embeddings"], 131072)
            self.assertEqual(json.loads((folder / "config.json").read_text()), original)
            verify_checkpoint(source)

    def test_flat_local_checkpoint_and_nested_hub_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flat = root / "local-model"
            self.checkpoint(flat)
            source, _ = discover_checkpoint(root, flat)
            self.assertEqual(source["folder"], str(flat))
            self.assertIsNone(source["revision"])
            snapshot = root / "hub" / MODEL_CACHE / "snapshots" / ("b"*40)
            self.checkpoint(snapshot)
            source, _ = discover_checkpoint(root)
            self.assertEqual(source["folder"], str(snapshot))

    def test_missing_shards_or_tokenizer_fail_without_downloading(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.checkpoint(folder)
            (folder / "model-00001.safetensors").unlink()
            with self.assertRaisesRegex(RuntimeError, "checkpoint file"):
                inspect_checkpoint(folder)
            self.checkpoint(folder)
            (folder / "tokenizer.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "tokenizer.json"):
                inspect_checkpoint(folder)

    def test_incompatible_config_and_ambiguous_snapshots_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_cache = root / MODEL_CACHE
            one, two = model_cache / "snapshots" / ("a"*40), model_cache / "snapshots" / ("b"*40)
            original = self.checkpoint(one)
            self.checkpoint(two)
            with self.assertRaisesRegex(RuntimeError, "unique snapshot"):
                discover_checkpoint(root)
            bad = {**original, "rope_scaling": {"rope_type": "linear", "factor": 2}}
            with self.assertRaisesRegex(RuntimeError, "different RoPE"):
                runtime_config(bad)
            with self.assertRaisesRegex(RuntimeError, "hidden_size"):
                runtime_config({**original, "hidden_size": 4096})

    def test_local_file_changes_are_detected_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.checkpoint(folder)
            source, _ = inspect_checkpoint(folder)
            (folder / "tokenizer.json").write_text('{"changed":true}')
            with self.assertRaisesRegex(RuntimeError, "changed since preparation"):
                verify_checkpoint(source)

    def test_missing_baseline_falls_back_but_explicit_bad_baseline_is_not_ignored(self):
        args = SimpleNamespace(model_dir=None, baseline_dir=None, cache_dir=Path("unused"))
        details = {"folder": "/local/checkpoint", "revision": "a"*40}
        with patch.object(runner, "baseline", side_effect=RuntimeError("missing historical artifacts")), \
                patch.object(runner, "discover_checkpoint", return_value=(details, {})) as locate:
            folder, info, _ = runner.model_setup(args)
            self.assertIsNone(folder)
            self.assertEqual(info["model_source"], "/local/checkpoint")
            args.baseline_dir = "/explicit/bad/baseline"
            with self.assertRaisesRegex(RuntimeError, "missing historical"):
                runner.model_setup(args)
            self.assertEqual(locate.call_count, 1)


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj, self.up_proj = nn.Linear(16, 32), nn.Linear(16, 32)
        self.down_proj = nn.Linear(32, 16)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj = [nn.Linear(16, 16) for _ in range(3)]

    def forward(self, x):
        return (self.q_proj(x) + self.k_proj(x) + self.v_proj(x)) / 3


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn, self.mlp = TinyAttention(), TinyMLP()

    def forward(self, x):
        return self.mlp(self.self_attn(x))


class Cache:
    def __init__(self):
        self.length = 0

    def get_seq_length(self):
        return self.length


class TinyQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyLayer(), TinyLayer()])
        self.embed, self.head = nn.Embedding(64, 16), nn.Linear(16, 64)
        self.generation_config = SimpleNamespace(eos_token_id=-1)
        self.to(torch.bfloat16)

    def forward(self, input_ids, past_key_values=None, **kwargs):
        x = self.embed(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        cache = past_key_values or Cache()
        cache.length += input_ids.shape[1]
        return SimpleNamespace(logits=self.head(x[:, -1:]), past_key_values=cache)


class Tokenizer:
    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))


class MetricTests(unittest.TestCase):
    def test_counts_against_independent_numpy_reference(self):
        rng = np.random.default_rng(42)
        values = rng.normal(size=(5, 515)).astype(np.float32)
        values[0] = 0
        values[1, :256] = 0
        values[2, :17] = 0
        values[3, :5] = [0, 1e-5, -1e-4, 1e-3, -1e-2]
        x = torch.tensor(values, dtype=torch.bfloat16)
        actual = tensor_metrics(x).numpy()
        reference = x.float().numpy()
        a = np.abs(reference)
        np.testing.assert_array_equal(actual[:, METRICS.index("zero_count")], (a == 0).sum(-1))
        np.testing.assert_allclose(actual[:, METRICS.index("mean_abs")], a.mean(-1), rtol=1e-5)
        np.testing.assert_allclose(actual[:, METRICS.index("rms")], np.sqrt((a*a).mean(-1)), rtol=1e-5)
        for threshold in THRESHOLDS:
            np.testing.assert_array_equal(actual[:, METRICS.index(f"small_{threshold:g}_count")], (a <= threshold).sum(-1))
        for block in BLOCKS:
            expected = (a[:, :515//block*block] == 0).reshape(5, 515//block, block).all(-1).sum(-1)
            np.testing.assert_array_equal(actual[:, METRICS.index(f"zero_blocks_{block}")], expected)
        histogram = actual[:, [i for i, name in enumerate(METRICS) if name.startswith("magnitude_bin_")]]
        np.testing.assert_array_equal(histogram.sum(-1), [515]*5)

    def test_nonfinite_is_not_counted_as_sparse(self):
        x = torch.tensor([[float("nan"), float("inf"), -float("inf"), 0, 1e-8]])
        row = tensor_metrics(x)[0].numpy()
        self.assertEqual(row[METRICS.index("nonfinite_count")], 3)
        self.assertEqual(row[METRICS.index("zero_count")], 1)
        self.assertEqual(sum(row[i] for i, key in enumerate(METRICS) if key.startswith("magnitude_bin")), 2)

    def test_hooks_are_read_only_and_release_handles(self):
        torch.manual_seed(12)
        model = TinyQwen()
        collector = Collector(model, 4, [1])
        ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        with torch.inference_mode():
            before = model(ids).logits
            collector.phase = "prefill"
            after = model(ids).logits
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            self.assertEqual(len(collector.pending), 2 * len(KINDS))
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "sample.npz"
                self.assertEqual(collector.save(path, 2, []), 2*len(KINDS)*4)
            collector.close()
            self.assertFalse(collector.handles)
            self.assertFalse(collector.pending)

    def test_weight_scan_is_exact(self):
        model = TinyQwen()
        with torch.no_grad():
            model.head.weight[:3].zero_()
        rows = scan_weights(model, lambda *args: None)
        row = next(row for row in rows if row["name"] == "head.weight")
        self.assertEqual(row["zero_count"], int((model.head.weight == 0).sum()))
        self.assertEqual(row["elements"], model.head.weight.numel())


class ProtocolTests(unittest.TestCase):
    def test_full_grid_and_positions(self):
        prompts = [{"index": i, "source_id": str(i), "domain": "test"} for i in range(300)]
        cases = make_cases(prompts)
        self.assertEqual(len(cases), 2100)
        self.assertEqual(len({c["id"] for c in cases}), 2100)
        self.assertEqual({c["input_tokens"] for c in cases}, set(LENGTHS))
        for length in (1, 2, 1024, 65536):
            values = positions(length)
            self.assertEqual(len(values), len(set(values)))
            self.assertEqual(values[-1], length-1)
            self.assertTrue(all(0 <= p < length for p in values))

    def test_real_context_crop_never_repeats_or_pads(self):
        self.assertEqual(context_slice(list(range(10)), 5), [0, 1, 2, 8, 9])
        self.assertEqual(context_slice(list(range(10)), 1), [0])
        self.assertEqual(context_slice(list(range(10)), 10), list(range(10)))
        with self.assertRaises(ValueError):
            context_slice([1, 2], 3)

    def test_stratified_selection_reproducible(self):
        rows = [{"domain": domain, "id": i} for domain in "abc" for i in range(10)]
        selected = stratified_order(rows)
        self.assertEqual(selected, stratified_order(rows))
        self.assertEqual([r["domain"] for r in selected[:6]], list("abcabc"))

    def test_resume_commit_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_tokens(folder / "tokens.gz", [1, 2, 3])
            self.assertEqual(read_tokens(folder / "tokens.gz"), [1, 2, 3])
            self.assertIsNone(complete_result(folder, "a"))
            row = {"identity": "a", "status": "pass", "artifacts": {"tokens.gz": sha(folder / "tokens.gz")}}
            atomic_json(folder / "result.json", row)
            self.assertEqual(complete_result(folder, "a"), row)
            with self.assertRaises(RuntimeError):
                complete_result(folder, "b")
            (folder / "tokens.gz").write_bytes(b"broken")
            with self.assertRaises(RuntimeError):
                complete_result(folder, "a")
            archive_attempt(folder)
            self.assertIsNone(complete_result(folder, "a"))
            self.assertEqual(len(list(folder.glob("attempt_*/result.json"))), 1)

    def test_prompt_balancing_and_cluster_uncertainty(self):
        a = prompt_distribution([0]*100)
        b = prompt_distribution([100])
        mean, _, _ = bootstrap_mean([a, b])
        self.assertEqual(mean[0], .5)
        self.assertEqual(mean[-1], .5)
        estimate, low, high = bootstrap_mean([[0], [1]], clusters=["same", "same"])
        self.assertEqual(estimate[0], .5)
        self.assertTrue(np.isnan(low[0]) and np.isnan(high[0]))
        mean, low, high = bootstrap_mean([[0], [0], [1]], clusters=["shared", "shared", "other"])
        self.assertAlmostEqual(mean[0], 1/3)
        self.assertTrue(0 <= low[0] <= mean[0] <= high[0] <= 1)


class GenerationIntegrationTests(unittest.TestCase):
    def test_controller_pause_resume_skips_saved_cases_and_recovers_interruption(self):
        """Exercise the real scheduler/checkpoints with explicit CPU test doubles."""
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp)
            original_tensor = torch.tensor
            def cpu_tensor(*args, **kwargs):
                kwargs.pop("device", None)
                return original_tensor(*args, **kwargs)
            def prompts(out, tokenizer, count, lengths, cache):
                folder = out / "prompts"
                folder.mkdir()
                result = []
                for i in range(count):
                    name = f"p{i:03d}.json.gz"
                    write_tokens(folder / name, {"prefix": [1], "context": [3]*max(lengths), "suffix": [2]})
                    result.append({"index": i, "source_id": str(i), "domain": "test", "tokens_file": name,
                                   "tokens_sha256": sha(folder / name), "context_sha256": str(i)})
                return result
            fake_transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: Tokenizer()))
            handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
            try:
                with patch.object(runner, "OUTPUTS", outputs), patch.object(runner, "environment", return_value={"cpu_test_double": True}), \
                        patch.object(runner, "baseline", return_value=(outputs, {"revision": "0"*40}, {"num_hidden_layers": 2})), \
                        patch.object(runner, "prepare_prompts", side_effect=prompts), \
                        patch.object(runner, "load_model", side_effect=lambda *args: TinyQwen()) as loading, \
                        patch.object(runner, "verify_hooks", return_value={"status": "cpu_test_double"}), \
                        patch.dict(sys.modules, {"transformers": fake_transformers}), patch("plot.plot"), \
                        patch("torch.tensor", side_effect=cpu_tensor), patch("torch.cuda.synchronize"), \
                        patch("torch.cuda.reset_peak_memory_stats"), patch("torch.cuda.max_memory_allocated", return_value=123), \
                        patch("torch.cuda.max_memory_reserved", return_value=456), patch("torch.cuda.empty_cache"):
                    with patch.object(sys, "argv", ["run.py", "--smoke", "--max-runs", "1"]):
                        self.assertEqual(runner.main(), 0)
                    out = next(outputs.glob("qwen_sparsity_smoke_*"))
                    manifest = json.loads((out / "manifest.json").read_text())
                    first = out / "cases" / manifest["cases"][0]["id"] / "result.json"
                    first_sha = sha(first)
                    resume = ["run.py", "--resume", str(out), "--max-runs", "1"]
                    with patch.object(sys, "argv", resume):
                        self.assertEqual(runner.main(), 0)
                    self.assertEqual(sha(first), first_sha)
                    self.assertEqual(sum(1 for p in (out / "cases").glob("*/result.json")), 2)
                    with patch.object(sys, "argv", resume), patch.object(runner, "execute", side_effect=KeyboardInterrupt("test pause")):
                        self.assertEqual(runner.main(), 130)
                    interrupted = out / "cases" / manifest["cases"][2]["id"]
                    self.assertEqual(json.loads((interrupted / "result.json").read_text())["status"], "interrupted")
                    with patch.object(sys, "argv", resume):
                        self.assertEqual(runner.main(), 0)
                    self.assertEqual(json.loads((interrupted / "result.json").read_text())["status"], "pass")
                    self.assertEqual(len(list(interrupted.glob("attempt_*/result.json"))), 1)
                    with patch.object(sys, "argv", resume):
                        self.assertEqual(runner.main(), 0)
                    before_loads = loading.call_count
                    with patch.object(sys, "argv", resume):
                        self.assertEqual(runner.main(), 0)
                    self.assertEqual(loading.call_count, before_loads)
                    self.assertTrue(json.loads((out / "summary.json").read_text())["all_pass"])
            finally:
                for signum, handler in handlers.items():
                    signal.signal(signum, handler)

    def test_generation_artifacts_resume_analysis_and_early_eos(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "prompts").mkdir()
            framing_data = {"prefix": [1], "context": list(range(2, 42)), "suffix": [42]}
            write_tokens(out / "prompts/p000.json.gz", framing_data)
            prompt = {"index": 0, "source_id": "test", "domain": "test", "tokens_file": "p000.json.gz", "context_sha256": "doc"}
            case = make_cases([prompt], [16])[0]
            manifest = {"out": str(out), "identity": "test", "prompts": [prompt], "cases": [case],
                        "config": {"num_hidden_layers": 2}, "protocol": {"max_new_tokens": 4, "block_definition": "test", "lengths": [16]}}
            atomic_json(out / "manifest.json", manifest)
            folder = out / "cases" / case["id"]
            folder.mkdir(parents=True)
            model, original_tensor = TinyQwen(), torch.tensor
            collector = Collector(model, 4, [1, 2, 3])
            def cpu_tensor(*args, **kwargs):
                kwargs.pop("device", None)
                return original_tensor(*args, **kwargs)
            with patch("torch.tensor", side_effect=cpu_tensor), patch("torch.cuda.synchronize"), \
                    patch("torch.cuda.reset_peak_memory_stats"), patch("torch.cuda.max_memory_allocated", return_value=123), \
                    patch("torch.cuda.max_memory_reserved", return_value=456):
                row = execute(model, Tokenizer(), collector, case, prompt, manifest, folder)
                self.assertEqual(row["generated_tokens"], 4)
                self.assertEqual(row["observations"], 2*len(KINDS)*7)
                self.assertEqual(complete_result(folder, "test")["status"], "pass")
                self.assertEqual(summarize(out, manifest)["pass"], 1)
                analyzed = analyze(out)
                self.assertEqual(len(analyzed[7]), 1)
                self.assertTrue((out / "distributions.csv").exists())
                # An early EOS must not invent unobserved decode samples.
                model.generation_config.eos_token_id = json.loads((folder / "generation.json").read_text())["token_ids"][0]
                second = out / "early_eos"
                second.mkdir()
                row = execute(model, Tokenizer(), collector, case, prompt, manifest, second)
                self.assertEqual(row["generated_tokens"], 1)
                self.assertEqual(row["observations"], 2*len(KINDS)*4)
            collector.close()


if __name__ == "__main__":
    unittest.main()
