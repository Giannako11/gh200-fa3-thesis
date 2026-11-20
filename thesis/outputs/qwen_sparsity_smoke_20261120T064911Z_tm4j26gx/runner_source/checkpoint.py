"""Locate an existing local Qwen checkpoint without historical run artifacts."""

import json
from pathlib import Path
import re

from protocol import sha

MODEL_CACHE = "models--Qwen--Qwen2.5-32B-Instruct"
EXPECTED = {"model_type": "qwen2", "hidden_size": 5120, "intermediate_size": 27648,
            "num_hidden_layers": 64, "num_attention_heads": 40,
            "num_key_value_heads": 8, "vocab_size": 152064}


def runtime_config(config):
    """Explicit YaRN4 experiment config, without editing the source config.json."""
    config = json.loads(json.dumps(config))
    for key, expected in EXPECTED.items():
        if config.get(key) != expected:
            raise RuntimeError(f"Not the expected Qwen2.5-32B checkpoint: {key}={config.get(key)!r}")
    if config.get("quantization_config") or config.get("torch_dtype", config.get("dtype")) != "bfloat16":
        raise RuntimeError("Expected the unquantized BF16 checkpoint.")
    rope = config.get("rope_scaling") or {}
    if rope and not (rope.get("rope_type", rope.get("type")) == "yarn"
                     and rope.get("factor") == 4.0
                     and rope.get("original_max_position_embeddings") == 32768):
        raise RuntimeError("Checkpoint has a different RoPE configuration; refusing to replace it silently.")
    if config.get("max_position_embeddings") not in (32768, 131072):
        raise RuntimeError("Unexpected checkpoint context configuration.")
    config["max_position_embeddings"] = 131072
    config["rope_scaling"] = {"rope_type": "yarn", "factor": 4.0,
                             "original_max_position_embeddings": 32768}
    return config


def file_record(path, hashed=False):
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing, empty or broken checkpoint file: {path}")
    record = {"name": path.name, "bytes": path.stat().st_size,
              "mtime_ns": path.stat().st_mtime_ns}
    if hashed:
        record["sha256"] = sha(path)
    return record


def inspect_checkpoint(folder):
    folder = Path(folder).expanduser().resolve()
    config = json.loads((folder / "config.json").read_text())
    effective = runtime_config(config)
    index = folder / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text()).get("weight_map", {})
        if not weight_map:
            raise RuntimeError(f"Empty safetensors weight map: {index}")
        shards = sorted(set(weight_map.values()))
    elif (folder / "model.safetensors").is_file():
        shards = ["model.safetensors"]
    else:
        raise RuntimeError(f"No safetensors checkpoint/index in {folder}")
    if any(not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors") for name in shards):
        raise RuntimeError("Unexpected shard names in checkpoint index.")
    records = [file_record(folder / name) for name in shards]
    # File checks do not pretend to validate tensor contents; the actual loader
    # validates the safetensors and model parameter shapes during GH200 smoke.
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        records.append(file_record(folder / name, hashed=True))
    for name in ("model.safetensors.index.json", "generation_config.json"):
        if (folder / name).is_file():
            records.append(file_record(folder / name, hashed=True))
    revision = folder.name if re.fullmatch(r"[0-9a-f]{40}", folder.name) and folder.parent.name == "snapshots" else None
    return {"folder": str(folder), "revision": revision, "files": records,
            "config_source": "local_checkpoint_with_explicit_runtime_yarn4",
            "baseline_success_claimed": False}, effective


def resolve_directory(folder):
    """Accept a checkpoint itself or a standard Hugging Face model cache root."""
    folder = Path(folder).expanduser().resolve()
    if (folder / "config.json").is_file():
        return inspect_checkpoint(folder)
    snapshots = folder / "snapshots"
    ref = folder / "refs/main"
    if ref.is_file():
        revision = ref.read_text().strip()
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise RuntimeError(f"Invalid local Hugging Face ref: {ref}")
        return inspect_checkpoint(snapshots / revision)
    candidates = sorted(p for p in snapshots.glob("*") if p.is_dir() and re.fullmatch(r"[0-9a-f]{40}", p.name))
    if len(candidates) != 1:
        raise RuntimeError(f"Cannot select a unique snapshot in {folder}; use --model-dir with the exact checkpoint directory.")
    return inspect_checkpoint(candidates[0])


def discover_checkpoint(cache, explicit=None):
    if explicit is not None:
        return resolve_directory(explicit)
    cache = Path(cache).expanduser().resolve()
    candidates = [cache / MODEL_CACHE, cache / "hub" / MODEL_CACHE,
                  cache / "Qwen2.5-32B-Instruct", cache / "Qwen" / "Qwen2.5-32B-Instruct"]
    if (cache / "config.json").is_file():
        candidates.insert(0, cache)
    found, errors = {}, []
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            source, config = resolve_directory(candidate)
            found[source["folder"]] = source, config
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(f"{candidate}: {exc}")
    if len(found) == 1:
        return next(iter(found.values()))
    if len(found) > 1:
        raise RuntimeError("Multiple local Qwen checkpoints found; use --model-dir:\n" + "\n".join(found))
    raise RuntimeError("No complete local Qwen checkpoint found. No model download attempted.\n"
                       + "Looked under: " + str(cache) + "\n"
                       + "Use --model-dir /path/to/directory/containing/config.json.\n"
                       + "\n".join(errors))


def verify_checkpoint(source):
    folder = Path(source["folder"])
    for prior in source["files"]:
        actual = file_record(folder / prior["name"], hashed="sha256" in prior)
        if actual != prior:
            raise RuntimeError(f"Local checkpoint changed since preparation: {folder / prior['name']}")
