"""Build reproducible real-text prompts; no padding, repeats or model downloads."""

from collections import Counter
import json
from pathlib import Path
import shutil
import time
import urllib.request

from protocol import (DATASET_BYTES, DATASET_SHA256, DATASET_URL, atomic_json,
                      sha, stratified_order, write_tokens)


def download_dataset(cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / "longbench-v2.json"
    if target.exists() and sha(target) == DATASET_SHA256:
        return target
    if target.exists():
        raise RuntimeError(f"Dataset checksum mismatch; inspect {target} before replacing it.")
    if shutil.disk_usage(cache).free < DATASET_BYTES * 2:
        raise RuntimeError("Need about 0.9 GiB free for the pinned prompt dataset download.")
    tmp = target.with_suffix(".part")
    print(f"Downloading pinned LongBench v2 prompts ({DATASET_BYTES / 2**20:.0f} MiB) once: {target}", flush=True)
    last = time.monotonic()
    with urllib.request.urlopen(DATASET_URL, timeout=60) as response, tmp.open("wb") as stream:
        size = 0
        while chunk := response.read(4 * 1024 * 1024):
            stream.write(chunk)
            size += len(chunk)
            if time.monotonic() - last > 10:
                print(f"Prompt download: {100 * size / DATASET_BYTES:.1f}%", flush=True)
                last = time.monotonic()
    if tmp.stat().st_size != DATASET_BYTES or sha(tmp) != DATASET_SHA256:
        raise RuntimeError(f"Incomplete/changed dataset download: {tmp}")
    tmp.replace(target)
    return target


def framing(tokenizer, item):
    # Tokenize the framing separately so exact token budgets preserve the question
    # and all chat delimiters. The context itself uses unchanged source token IDs.
    marker = "__THESIS_CONTEXT_BOUNDARY_711__"
    choices = "\n".join(f"{letter}. {item['choice_' + letter]}" for letter in "ABCD")
    user = ("Read the following context and answer the question. Explain your reasoning briefly, "
            "then state the selected option.\n\nContext:\n" + marker
            + "\n\nQuestion:\n" + item["question"] + "\n" + choices)
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    if text.count(marker) != 1:
        raise RuntimeError("Unexpected tokenizer chat template.")
    prefix, suffix = text.split(marker)
    return (tokenizer.encode(prefix, add_special_tokens=False),
            tokenizer.encode(suffix, add_special_tokens=False))


def context_slice(ids, count):
    """Head+tail cropping; never repeat or pad text to reach a requested length."""
    if count <= 0 or len(ids) < count:
        raise ValueError("Insufficient real context for requested input length.")
    head = (count + 1) // 2
    tail = count - head
    return ids[:head] + (ids[-tail:] if tail else [])


def prepare_prompts(out, tokenizer, count, lengths, cache):
    target = Path(out) / "prompts"
    target.mkdir(exist_ok=True)
    source = download_dataset(cache)
    rows = json.loads(source.read_text(encoding="utf-8"))
    if len({item["_id"] for item in rows}) != len(rows):
        raise RuntimeError("Duplicate prompt IDs in source dataset.")
    # Questions are distinct. Shared source contexts remain explicitly labelled
    # so uncertainty can resample documents instead of treating them as independent.
    import hashlib
    chosen = []
    for inspected, item in enumerate(stratified_order(rows), 1):
        prefix, suffix = framing(tokenizer, item)
        budget = max(lengths) - len(prefix) - len(suffix)
        if budget <= 0:
            continue
        # Cap parsing to the largest requested input. This is a genuine context
        # excerpt; no fabricated filler is introduced for a short source.
        context = tokenizer(item["context"], add_special_tokens=False,
                            truncation=True, max_length=max(lengths))["input_ids"]
        if len(context) < budget or min(lengths) <= len(prefix) + len(suffix):
            continue
        index = len(chosen)
        name = f"p{index:03d}.json.gz"
        write_tokens(target / name, {"prefix": prefix, "context": context, "suffix": suffix})
        chosen.append({"index": index, "source_id": item["_id"],
                       "context_sha256": hashlib.sha256(item["context"].encode()).hexdigest(),
                       "domain": item["domain"], "sub_domain": item["sub_domain"],
                       "source_length_group": item["length"], "tokens_file": name,
                       "tokens_sha256": sha(target / name), "context_tokens_kept": len(context),
                       "framing_tokens": len(prefix) + len(suffix)})
        if len(chosen) % 10 == 0 or len(chosen) == count:
            print(f"Prepared prompts: {len(chosen)}/{count}; inspected {inspected} source items", flush=True)
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise RuntimeError(f"Only {len(chosen)} distinct source questions are long enough; need {count}. "
                           "No duplicated/padded prompts were substituted.")
    atomic_json(target / "index.json", {"source_url": DATASET_URL, "source_sha256": DATASET_SHA256,
                "domains": dict(Counter(p["domain"] for p in chosen)), "prompts": chosen,
                "selection": "seed-711 domain round-robin, distinct question IDs, long-enough only",
                "distinct_source_contexts": len({p["context_sha256"] for p in chosen}),
                "context_policy": "first max-T source tokens, then head+tail crop per exact input budget",
                "scope": "long-source LongBench-v2 subset; not the population of all Qwen workloads",
                "quality_note": "Cropping may remove answer evidence. This is not a task-accuracy evaluation."})
    return chosen
