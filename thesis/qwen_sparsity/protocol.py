"""Immutable experiment settings and CPU-only persistence helpers."""

from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import random

MODEL = "Qwen/Qwen2.5-32B-Instruct"
LENGTHS = (1024, 2048, 4096, 8192, 16384, 32768, 65536)
PROMPTS = 300
MAX_NEW_TOKENS = 64
PREFILL_ROWS = 16
DECODE_STEPS = (1, 2, 4, 8, 16, 32, 48, 63)
THRESHOLDS = (1e-5, 1e-4, 1e-3, 1e-2)
BLOCKS = (16, 64, 256)
MAG_EDGES = (1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
KINDS = ("mlp_input", "mlp_down_input", "attention_q", "attention_k", "attention_v")
DATASET_REVISION = "ac35867b9c9436dc81c4c7535fe640ba7cf47877"
DATASET_URL = ("https://huggingface.co/datasets/zai-org/LongBench-v2/resolve/"
               + DATASET_REVISION + "/data.json")
DATASET_BYTES = 465490535
DATASET_SHA256 = "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2"
METRICS = ("zero_count", "nonfinite_count", "mean_abs", "rms", "max_abs",
           *(f"small_{t:g}_count" for t in THRESHOLDS),
           *(f"zero_blocks_{b}" for b in BLOCKS),
           *(f"magnitude_bin_{i}" for i in range(len(MAG_EDGES) + 1)))


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _hash(stream)


def _hash(stream):
    value = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        value.update(chunk)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def write_tokens(path, ids):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as stream:
        json.dump(ids, stream, separators=(",", ":"))
    with tmp.open("rb") as stream:
        os.fsync(stream.fileno())
    tmp.replace(path)


def read_tokens(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def positions(length, count=PREFILL_ROWS):
    """Unique, evenly spaced token positions including both endpoints."""
    count = min(count, length)
    if count == 1:
        return [length - 1]
    return [i * (length - 1) // (count - 1) for i in range(count)]


def make_cases(prompts, lengths=LENGTHS):
    # Paired lengths: the same source prompt appears at every context length.
    return [{"id": f"p{p['index']:03d}_t{t:05d}", "prompt_index": p["index"],
             "source_id": p["source_id"], "domain": p["domain"], "input_tokens": t}
            for p in prompts for t in lengths]


def stratified_order(rows, seed=711):
    """Round-robin domains, with deterministic shuffled order inside each."""
    groups = {}
    for row in rows:
        groups.setdefault(row["domain"], []).append(row)
    rng = random.Random(seed)
    for domain in sorted(groups):
        rng.shuffle(groups[domain])
    result = []
    while any(groups.values()):
        for domain in sorted(groups):
            if groups[domain]:
                result.append(groups[domain].pop())
    return result


def complete_result(folder, identity, case=None):
    """Only a committed success with matching artifacts counts as complete."""
    folder = Path(folder)
    path = folder / "result.json"
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    if row.get("identity") != identity:
        raise RuntimeError(f"Protocol mismatch: {path}")
    if case is not None and any(row.get(k) != v for k, v in case.items()):
        raise RuntimeError(f"Result/configuration mismatch: {path}")
    if row.get("status") != "pass":
        return None
    for name, checksum in row["artifacts"].items():
        if sha(folder / name) != checksum:
            raise RuntimeError(f"Changed or incomplete artifact: {folder / name}")
    return row


def archive_attempt(folder):
    """Keep every partial/failed attempt without mixing it into a retry."""
    folder = Path(folder)
    files = [p for p in folder.iterdir() if p.is_file()]
    if files:
        import time
        history = folder / ("attempt_" + str(time.time_ns()))
        history.mkdir()
        for path in files:
            path.replace(history / path.name)
