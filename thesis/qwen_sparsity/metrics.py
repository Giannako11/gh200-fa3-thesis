"""Read-only activation hooks with bounded GPU workspace and compact summaries."""

import os
from pathlib import Path

from protocol import BLOCKS, KINDS, MAG_EDGES, METRICS, THRESHOLDS, positions


def tensor_metrics(x):
    """One observation per token row. Exact zeros and thresholds stay separate."""
    import torch
    a = x.detach().float().abs()
    finite = torch.isfinite(a)
    clean = torch.where(finite, a, 0)
    zero = a == 0
    values = [zero.sum(-1), (~finite).sum(-1), clean.mean(-1),
              clean.square().mean(-1).sqrt(), clean.amax(-1)]
    values.extend(((a <= threshold) & finite).sum(-1) for threshold in THRESHOLDS)
    width = x.shape[-1]
    for block in BLOCKS:
        full = width // block
        values.append(zero[:, :full * block].reshape(x.shape[0], full, block).all(-1).sum(-1)
                      if full else torch.zeros(x.shape[0], device=x.device))
    # Bin 0 includes exact zeros. Histogram mass excludes NaN/Inf, which are
    # recorded separately and make the case fail validation.
    previous = torch.zeros_like(a, dtype=torch.bool)
    for edge in MAG_EDGES:
        current = (a <= edge) & finite
        values.append((current & ~previous).sum(-1))
        previous = current
    values.append((finite & ~previous).sum(-1))
    return torch.stack(values, dim=-1).float()


class Collector:
    def __init__(self, model, prefill_rows, decode_steps):
        self.pending = []
        self.handles = []
        self.phase = None
        self.step = 0
        self.input_tokens = 0
        self.prefill_rows = prefill_rows
        self.decode_steps = tuple(decode_steps)
        self.module_count = 0
        for layer, module in enumerate(model.model.layers):
            self.handles.append(module.mlp.register_forward_pre_hook(self._pre(layer, "mlp_input")))
            self.handles.append(module.mlp.down_proj.register_forward_pre_hook(self._pre(layer, "mlp_down_input")))
            for projection in "qkv":
                self.handles.append(getattr(module.self_attn, projection + "_proj").register_forward_hook(
                    self._post(layer, "attention_" + projection)))
            self.module_count += len(KINDS)

    def _pre(self, layer, kind):
        def hook(module, args):
            self.observe(args[0], layer, kind)
            return None
        return hook

    def _post(self, layer, kind):
        def hook(module, args, output):
            self.observe(output, layer, kind)
            return None
        return hook

    def observe(self, x, layer, kind):
        if self.phase is None or (self.phase == "decode" and self.step not in self.decode_steps):
            return
        import torch
        if x.ndim != 3 or x.shape[0] != 1:
            raise RuntimeError(f"Unexpected {kind} shape: {tuple(x.shape)}")
        row_ids = positions(x.shape[1], self.prefill_rows) if self.phase == "prefill" else [0]
        selected = x[0].index_select(0, torch.tensor(row_ids, device=x.device))
        measured = tensor_metrics(selected)
        absolute = row_ids if self.phase == "prefill" else [self.input_tokens + self.step - 1]
        # No original activation is retained, only small detached metrics.
        self.pending.append((layer, KINDS.index(kind), int(self.phase == "decode"),
                             self.step, absolute, list(x.shape), str(x.dtype), measured))

    def clear(self):
        self.pending.clear()
        self.phase = None

    def save(self, path, expected_layers, sampled_decode_steps):
        import numpy as np
        import torch
        expected = expected_layers * len(KINDS) * (1 + len(sampled_decode_steps))
        if len(self.pending) != expected:
            raise RuntimeError(f"Hook coverage mismatch: {len(self.pending)} calls, expected {expected}")
        arrays, metadata = [], []
        seen = set()
        for layer, kind, phase, step, token_ids, shape, dtype, measured in self.pending:
            key = (layer, kind, phase, step)
            if key in seen:
                raise RuntimeError("Duplicate hook observation")
            seen.add(key)
            arrays.append(measured)
            for token in token_ids:
                metadata.append((layer, kind, phase, step, token, shape[-1], shape[1]))
            if dtype != "torch.bfloat16":
                raise RuntimeError(f"Unexpected activation dtype: {dtype}")
        data = torch.cat(arrays, dim=0).cpu().numpy()
        if not np.isfinite(data).all() or np.any(data[:, METRICS.index("nonfinite_count")] != 0):
            raise RuntimeError("Non-finite sampled activations or metrics")
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as stream:
            np.savez_compressed(stream, metrics=data, metadata=np.asarray(metadata, dtype=np.int32),
                                metric_names=np.asarray(METRICS), kinds=np.asarray(KINDS),
                                metadata_names=np.asarray(("layer", "kind", "phase", "decode_step",
                                                           "token_position", "width", "sequence_rows")))
            stream.flush()
            os.fsync(stream.fileno())
        tmp.replace(path)
        return len(metadata)

    def close(self):
        self.clear()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def scan_weights(model, progress):
    """Exact whole-weight zero/finite counts, streamed in <=4M-element chunks."""
    import torch
    rows = []
    parameters = [(name, p) for name, p in model.named_parameters() if p.ndim == 2]
    for index, (name, weight) in enumerate(parameters, 1):
        flat = weight.detach().view(-1)
        counts = torch.zeros(2, dtype=torch.int64, device=weight.device)
        for start in range(0, flat.numel(), 4 * 1024 * 1024):
            part = flat[start:start + 4 * 1024 * 1024]
            counts += torch.stack(((part == 0).sum(), (~torch.isfinite(part)).sum()))
        zero, bad = counts.cpu().tolist()
        if bad:
            raise RuntimeError(f"Non-finite weights: {name}")
        rows.append({"name": name, "shape": list(weight.shape), "dtype": str(weight.dtype),
                     "elements": weight.numel(), "zero_count": zero, "nonfinite_count": bad,
                     "sparsity_pct": 100 * zero / weight.numel()})
        if index % 32 == 0 or index == len(parameters):
            progress(index, len(parameters), "scan static weights")
    return rows
