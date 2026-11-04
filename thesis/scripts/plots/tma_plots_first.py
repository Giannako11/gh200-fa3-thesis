#!/usr/bin/env python3
"""Make 39 original TMA PNGs, or seven PNGs with --malloc-bound-only.

Usage: python thesis/scripts/plots/tma_plots_first.py PATH/TO/results.csv
Dependencies: matplotlib, numpy. No CUDA or GPU is needed.
"""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import math
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np


SIZES = [2**power for power in range(12, 35)]
TILES = [(32, 32), (32, 128), (64, 64), (128, 32), (64, 128), (128, 64), (128, 128)]
SOURCES = ["hbm", "pinned", "registered", "malloc"]
KERNELS = ["loads_direct", "loads_shared", "tma_shared"]
COLORS = ["#0072B2", "#E69F00", "#009E73"]
LABELS = ["Regular loads: direct", "Regular loads: shared", "TMA: shared"]
SOURCE_LABELS = {"hbm": "HBM input", "pinned": "CUDA mapped host input",
                 "registered": "Registered host input", "malloc": "Plain malloc input",
                 "malloc_bound": "CPU-bound system input (posix_memalign + mbind)"}
STATUSES = {"ok", "error", "not_run", "device_fault", "worker_error"}
FIGURE_COUNT = 39


@dataclass
class Row:
    status: str
    ms: float | None
    resident: int


def size_label(n, short=False):
    for unit, scale in [("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)]:
        if n >= scale:
            return f"{n / scale:g}{unit[0]}" if short else f"{n / scale:g} {unit}"


def parse_size(text):
    for unit, scale in [("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)]:
        if text.endswith(unit):
            try:
                return int(text[:-len(unit)]) * scale
            except ValueError:
                break
    raise argparse.ArgumentTypeError("Use an integer size such as 1MiB or 64KiB")


def parse_tile(text):
    try:
        tile = tuple(int(part) for part in text.split("x"))
    except ValueError:
        tile = ()
    if tile not in TILES:
        raise argparse.ArgumentTypeError("Choose one of the seven sweep tiles, e.g. 32x32")
    return tile


def valid_pair(n, tile):
    tile_bytes = tile[0] * tile[1] * 4
    return n >= tile_bytes and n % tile_bytes == 0


def source_selections():
    singles = [[source] for source in SOURCES]
    return singles + ([SOURCES] if len(SOURCES) > 1 else [])


# Κρατάμε κάθε μέτρηση χωριστά. Δεν βγάζουμε μέσο όρο από διαφορετικά runs.
def read_results(path):
    data, thread_counts = {}, set()
    required = {"input_bytes", "tile_rows", "tile_cols", "tile_bytes", "source",
                "kernel", "gpu_threads", "status", "kernel_ms", "cpu_residency_verified"}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV is missing columns: {sorted(required - set(reader.fieldnames or []))}")
        for line, raw in enumerate(reader, 2):
            try:
                n = int(raw["input_bytes"])
                tile = int(raw["tile_rows"]), int(raw["tile_cols"])
                source, kernel, status = raw["source"], raw["kernel"], raw["status"]
                threads, resident = int(raw["gpu_threads"]), int(raw["cpu_residency_verified"])
                if (n not in SIZES or tile not in TILES or not valid_pair(n, tile)
                        or int(raw["tile_bytes"]) != tile[0] * tile[1] * 4
                        or source not in SOURCES or kernel not in KERNELS or status not in STATUSES
                        or threads not in (64, 128, 256)
                        or resident not in ({-1} if source == "hbm" else {0, 1})):
                    raise ValueError("unsupported configuration or status")
                ms = float(raw["kernel_ms"]) if status == "ok" else None
                if ms is not None and (not math.isfinite(ms) or ms <= 0):
                    raise ValueError("successful timing must be finite and positive")
                key = n, tile, source, kernel
                if key in data:
                    raise ValueError(f"duplicate measurement: {key}")
                data[key] = Row(status, ms, resident)
                thread_counts.add(threads)
            except (ValueError, TypeError) as error:
                raise ValueError(f"CSV line {line}: {error}") from error
    if not data:
        raise ValueError("CSV has no measurements")
    if len(thread_counts) != 1:
        raise ValueError("Mixed GPU thread counts: use a separate CSV for each setting")
    return data, thread_counts.pop()


def timing(data, n, tile, source, kernel):
    row = data.get((n, tile, source, kernel))
    if row is None or row.status != "ok":
        return math.nan, False
    return row.ms, source != "hbm" and row.resident != 1


# Υπολογίζουμε GB/s από τα bytes εισόδου και τον συνολικό χρόνο του kernel.
def input_rate(data, n, tile, source, kernel):
    ms, unverified = timing(data, n, tile, source, kernel)
    if not math.isfinite(ms):
        return math.nan, False
    rate = n / (ms * 1e6)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Non-finite or non-positive effective input rate")
    return rate, unverified


# Ο λόγος υπάρχει μόνο όταν και οι δύο μετρήσεις έχουν πετύχει.
def ratio(data, n, tile, numerator_source, numerator_kernel, denominator_source, denominator_kernel):
    a, unverified_a = timing(data, n, tile, numerator_source, numerator_kernel)
    b, unverified_b = timing(data, n, tile, denominator_source, denominator_kernel)
    if not math.isfinite(a) or not math.isfinite(b):
        return math.nan, False
    result = a / b
    if not math.isfinite(result) or result <= 0:
        raise ValueError("Non-finite or non-positive timing ratio")
    return result, unverified_a or unverified_b


class Figures:
    def __init__(self, output, dpi, note):
        self.output, self.dpi, self.note = output, dpi, note
        self.index = []

    def save(self, fig, group, name, title, handles, labels, footnote, heatmap=False):
        fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)
        if handles:
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.935),
                       ncol=len(labels), frameon=False)
        fig.text(0.5, 0.015, footnote + "\n" + self.note, ha="center", va="bottom", fontsize=8)
        if heatmap:
            fig.subplots_adjust(left=0.08, right=0.88, bottom=0.20, top=0.84,
                                wspace=0.22, hspace=0.55)
        else:
            fig.tight_layout(rect=(0, 0.10, 1, 0.87))
        fig.savefig(self.output / f"{name}.png", dpi=self.dpi, facecolor="white", format="png")
        plt.close(fig)
        self.index.append((group, f"{name}.png", title))
        done = len(self.index)
        bar = "#" * (done * 24 // FIGURE_COUNT)
        print(f"[{bar:<24}] {done:2d}/{FIGURE_COUNT} {name}", flush=True)


def axes_for(count, heatmap=False):
    if count == 1:
        fig, axis = plt.subplots(figsize=(13 if heatmap else 10, 5.6))
        return fig, [axis]
    fig, axes = plt.subplots(2, 2, figsize=(21 if heatmap else 16, 10.5))
    flat = list(axes.flat)
    for unused in flat[count:]:
        unused.set_visible(False)
    return fig, flat[:count]


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)


def legend_lines(labels, colors):
    return [Line2D([], [], color=color, marker="o", label=label)
            for label, color in zip(labels, colors)]


# Οι μπάρες ξεκινούν από το μηδέν. Κενή μέτρηση δεν γίνεται μπάρα μηδενικού χρόνου.
def draw_bars(ax, data, cases, source, category_labels, scale, unit, upper):
    x = np.arange(len(cases))
    width = 0.24
    for k, kernel in enumerate(KERNELS):
        points = [timing(data, n, tile, source, kernel) for n, tile in cases]
        values = [ms * scale for ms, _ in points]
        centers = x + (k - 1) * width
        bars = ax.bar(centers, values, width, color=COLORS[k], label=LABELS[k])
        for position, patch, (ms, unverified) in zip(centers, bars, points):
            if not math.isfinite(ms):
                ax.text(position, 0.025, "×", transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", color=COLORS[k], fontsize=11)
            elif unverified:
                patch.set_hatch("///")
                patch.set_edgecolor("#333333")
                patch.set_linewidth(0.4)
    ax.set_xticks(x, category_labels, rotation=35, ha="right")
    ax.set_ylim(0, upper)
    ax.set_ylabel(f"Kernel time ({unit})")
    ax.set_title(SOURCE_LABELS[source])
    style_axis(ax)


def bar_figures(data, writer, tile, fixed_size):
    groups = [("KiB", [n for n in SIZES if n < 2**20]),
              ("MiB", [n for n in SIZES if 2**20 <= n < 2**30]),
              ("GiB", [n for n in SIZES if n >= 2**30])]
    jobs = [(unit, [(n, tile) for n in sizes], [size_label(n) for n in sizes],
             f"Input sizes in {unit} | tile {tile[0]}×{tile[1]}") for unit, sizes in groups]
    jobs.append(("tiles_" + size_label(fixed_size).replace(" ", ""),
                 [(fixed_size, t) for t in TILES], [f"{h}×{w}" for h, w in TILES],
                 f"Tile comparison | input {size_label(fixed_size)}"))
    for name, cases, labels, title in jobs:
        values = [timing(data, n, t, s, k)[0] for n, t in cases for s in SOURCES for k in KERNELS]
        finite = [v for v in values if math.isfinite(v)]
        scale, unit = (1, "ms") if finite and max(finite) >= 1 else (1000, "µs")
        upper = max(finite) * scale * 1.18 if finite else 1
        for selected in source_selections():
            fig, axes = axes_for(len(selected))
            for ax, source in zip(axes, selected):
                draw_bars(ax, data, cases, source, labels, scale, unit, upper)
                ax.set_xlabel("Tile (rows × columns)" if name.startswith("tiles_") else "Input size")
            suffix = selected[0] if len(selected) == 1 else "all_sources"
            handles = [Rectangle((0, 0), 1, 1, color=color) for color in COLORS]
            writer.save(fig, "barplots", f"time_{name}_{suffix}", title, handles, LABELS,
                        "× = failed/missing or tile does not fit; hatching = host placement unverified. Same y-scale across allocations.")


def draw_curves(ax, series, labels, colors, ylabel, limits, ratio_axis=True):
    for points, label, color in zip(series, labels, colors):
        values = np.array([value for value, _ in points])
        unverified = np.array([flag for _, flag in points])
        good = np.isfinite(values)
        ax.plot(SIZES, values, color=color, linewidth=1.6, label=label)
        for subset, face in [(good & ~unverified, color), (good & unverified, "white")]:
            ax.scatter(np.array(SIZES)[subset], values[subset], facecolors=face,
                       edgecolors=color, s=26, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_ylim(*limits)
    if ratio_axis:
        ax.axhline(1, color="#555555", linestyle="--", linewidth=1)
        ax.set_yscale("log", base=2)
        lo, hi = np.log2(limits)
        step = 0.25 if hi-lo <= 2 else (0.5 if hi-lo <= 4 else math.ceil((hi-lo)/6))
        ax.set_yticks(2.0 ** np.arange(math.ceil(lo/step)*step, hi + 1e-9, step))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3g}×"))
    ax.set_xticks(SIZES, [size_label(n, short=True) for n in SIZES], rotation=60, ha="right", fontsize=8)
    ax.set_xlabel("Input size (K = KiB, M = MiB, G = GiB)")
    ax.set_ylabel(ylabel)
    style_axis(ax)


def ratio_limits(all_series):
    values = [v for series in all_series for v, _ in series if math.isfinite(v)] + [1]
    return min(values) / 1.3, max(values) * 1.3


def speedup_figures(data, writer, tile):
    labels = ["Regular shared / TMA", "Regular direct / TMA"]
    points = {s: [[ratio(data, n, tile, s, k, s, "tma_shared") for n in SIZES]
                  for k in ["loads_shared", "loads_direct"]] for s in SOURCES}
    limits = ratio_limits([series for source in SOURCES for series in points[source]])
    for selected in source_selections():
        fig, axes = axes_for(len(selected))
        for ax, source in zip(axes, selected):
            draw_curves(ax, points[source], labels, COLORS[:2], "TMA speedup (higher is faster)", limits)
            ax.set_title(SOURCE_LABELS[source])
        suffix = selected[0] if len(selected) == 1 else "all_sources"
        writer.save(fig, "speedup", f"speedup_{suffix}", f"TMA speedup | tile {tile[0]}×{tile[1]}",
                    legend_lines(labels, COLORS[:2]), labels,
                    "Above 1×: TMA is faster. Open markers: host placement unverified. Gaps: failed/missing timings.")


def heatmap_figures(data, writer):
    values, flags = {}, {}
    for source in SOURCES:
        values[source] = np.full((len(TILES), len(SIZES)), np.nan)
        flags[source] = np.zeros((len(TILES), len(SIZES)), dtype=bool)
        for j, tile in enumerate(TILES):
            for i, n in enumerate(SIZES):
                if valid_pair(n, tile):
                    value, flag = ratio(data, n, tile, source, "loads_shared", source, "tma_shared")
                    if math.isfinite(value):
                        values[source][j, i] = math.log2(value)
                        flags[source][j, i] = flag
    finite = [abs(v) for array in values.values() for v in array.flat if math.isfinite(v)]
    limit = max([1] + finite)
    norm = Normalize(-limit, limit)
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#eeeeee")
    for selected in source_selections():
        fig, axes = axes_for(len(selected), heatmap=True)
        for ax, source in zip(axes, selected):
            image = ax.imshow(np.ma.masked_invalid(values[source]), cmap=cmap, norm=norm, aspect="auto")
            for j, tile in enumerate(TILES):
                for i, n in enumerate(SIZES):
                    value = values[source][j, i]
                    if math.isfinite(value):
                        if flags[source][j, i]:
                            ax.add_patch(Rectangle((i-.5, j-.5), 1, 1, fill=False,
                                                   hatch="///", edgecolor="#777777", linewidth=0))
                        ax.text(i, j, f"{2**value:.2g}", ha="center", va="center", fontsize=7)
                    elif valid_pair(n, tile):
                        ax.text(i, j, "×", ha="center", va="center", color="#666666")
            ax.set_xticks(range(len(SIZES)), [size_label(n, short=True) for n in SIZES],
                          rotation=60, ha="right", fontsize=8)
            ax.set_yticks(range(len(TILES)), [f"{h}×{w}" for h, w in TILES])
            ax.set_xlabel("Input size (K = KiB, M = MiB, G = GiB)")
            ax.set_ylabel("Tile (rows × columns)")
            ax.set_title(SOURCE_LABELS[source])
        color_ax = fig.add_axes((0.92, 0.25, 0.015, 0.45))
        colorbar = fig.colorbar(image, cax=color_ax, ticks=np.linspace(-limit, limit, 5))
        colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{2**v:.2g}×"))
        colorbar.set_label("Regular shared / TMA")
        suffix = selected[0] if len(selected) == 1 else "all_sources"
        writer.save(fig, "heatmaps", f"tiles_speedup_{suffix}", "TMA speedup across input sizes and tiles",
                    [], [], "Cells show speedup. Green > 1×; red < 1×. Grey: tile does not fit; ×: failed/missing. Hatching: placement unverified.", True)


def host_hbm_figures(data, writer, tile):
    labels = ["Pinned / HBM", "Registered / HBM", "Malloc / HBM"]
    points = {k: [[ratio(data, n, tile, s, k, "hbm", k) for n in SIZES]
                  for s in SOURCES[1:]] for k in KERNELS}
    limits = ratio_limits([series for kernel in KERNELS for series in points[kernel]])
    for selected in [[k] for k in KERNELS] + [KERNELS]:
        fig, axes = axes_for(len(selected))
        for ax, kernel in zip(axes, selected):
            draw_curves(ax, points[kernel], labels, COLORS, "Host / HBM kernel time", limits)
            ax.set_title(LABELS[KERNELS.index(kernel)])
        suffix = selected[0] if len(selected) == 1 else "all_kernels"
        writer.save(fig, "host_vs_hbm", f"host_vs_hbm_{suffix}", f"Host input relative to HBM | tile {tile[0]}×{tile[1]}",
                    legend_lines(labels, COLORS), labels,
                    "Above 1×: host input is slower. Open markers: host placement unverified. Gaps: failed/missing timings.")


def input_rate_figures(data, writer, tile):
    points = {s: [[input_rate(data, n, tile, s, k) for n in SIZES]
                  for k in KERNELS] for s in SOURCES}
    finite = [v for source in SOURCES for series in points[source]
              for v, _ in series if math.isfinite(v)]
    limits = 0, max(finite, default=1) * 1.15
    for selected in source_selections():
        fig, axes = axes_for(len(selected))
        for ax, source in zip(axes, selected):
            draw_curves(ax, points[source], LABELS, COLORS, "Effective input rate (GB/s)",
                        limits, ratio_axis=False)
            ax.set_title(SOURCE_LABELS[source])
        suffix = selected[0] if len(selected) == 1 else "all_sources"
        writer.save(fig, "input_rate", f"input_rate_{suffix}",
                    f"Effective input processing rate | tile {tile[0]}×{tile[1]}",
                    legend_lines(LABELS, COLORS), LABELS,
                    "Input bytes / complete kernel time, including HBM output stores. Open markers: placement unverified. Gaps: failed/missing.")


# Βοηθητικό αρχείο για εμάς. Δεν δημιουργούμε figure ορθότητας.
def write_checks(data, output):
    expected = sum(valid_pair(n, tile) for n in SIZES for tile in TILES)
    with (output / "checks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "kernel", "expected", "recorded", "ok", "failed_or_not_run",
                         "missing", "host_placement_verified", "host_placement_unverified"])
        for source in SOURCES:
            for kernel in KERNELS:
                rows = [r for (_, _, s, k), r in data.items() if s == source and k == kernel]
                good = [r for r in rows if r.status == "ok"]
                verified = sum(r.resident == 1 for r in good) if source != "hbm" else ""
                unverified = sum(r.resident != 1 for r in good) if source != "hbm" else ""
                writer.writerow([source, kernel, expected, len(rows), len(good), len(rows)-len(good),
                                 expected-len(rows), verified, unverified])


# Νέος φάκελος για κάθε εκτέλεση, ώστε να μη σβήνονται παλιότερα plots.
def create_output(output_dir=None):
    if output_dir is None:
        try:
            date = datetime.now(ZoneInfo("Europe/Athens")).strftime("%d_%m_%y")
        except ZoneInfoNotFoundError as error:
            raise ValueError("Athens timezone data is missing; install tzdata or use --output-dir") from error
        thesis = Path(__file__).resolve().parents[2]
        output_dir = thesis / "outputs" / "plots" / f"{Path(__file__).stem}_plots_{date}"
    base = output_dir.resolve()
    base.parent.mkdir(parents=True, exist_ok=True)
    output, attempt = base, 1
    while True:
        try:
            output.mkdir()
            return output
        except FileExistsError:
            attempt += 1
            output = base.with_name(f"{base.name}_{attempt:02d}")


def main():
    global SOURCES, FIGURE_COUNT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, help="One sweep's results.csv")
    parser.add_argument("--output-dir", type=Path,
                        help="Override thesis/outputs/plots/tma_plots_first_plots_DD_MM_YY")
    parser.add_argument("--tile", type=parse_tile, default=(32, 32), help="Fixed tile for size plots (default: 32x32)")
    parser.add_argument("--fixed-size", type=parse_size, default=2**20, help="Input for tile bars (default: 1MiB)")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--malloc-bound-only", action="store_true",
                        help="Plot one malloc_bound sweep as seven PNGs; preserve its distinct allocation label")
    args = parser.parse_args()
    SOURCES = ["hbm", "pinned", "registered", "malloc"]
    FIGURE_COUNT = 39
    if args.malloc_bound_only:
        SOURCES = ["malloc_bound"]
        FIGURE_COUNT = 7
    if args.fixed_size not in SIZES or not all(valid_pair(args.fixed_size, t) for t in TILES):
        parser.error("--fixed-size must be a sweep size from 64KiB to 16GiB so all tiles fit")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    try:
        data, threads = read_results(args.csv)
        expected = sum(valid_pair(n, t) for n in SIZES for t in TILES) * len(SOURCES) * len(KERNELS)
        passed = sum(row.status == "ok" for row in data.values())
        if not passed:
            raise ValueError("CSV has no successful timings to plot")
        output = create_output(args.output_dir)
        checks = output / "_checks"
        checks.mkdir()
        print(f"Outputs: {output}", flush=True)
        note = (f"One timed launch per case; {threads} GPU threads/block; no error bars. "
                f"Recorded {len(data)}/{expected}; successful {passed}.")
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                             "axes.titlesize": 11, "axes.labelsize": 10})
        writer = Figures(output, args.dpi, note)
        write_checks(data, checks)
        bar_figures(data, writer, args.tile, args.fixed_size)
        speedup_figures(data, writer, args.tile)
        heatmap_figures(data, writer)
        if not args.malloc_bound_only:
            host_hbm_figures(data, writer, args.tile)
        input_rate_figures(data, writer, args.tile)
        with (checks / "plots_index.csv").open("w", newline="", encoding="utf-8") as handle:
            index = csv.writer(handle)
            index.writerow(["group", "filename", "title"])
            index.writerows(writer.index)
        notes = [f"Input: {args.csv.resolve()}", f"SHA256: {hashlib.sha256(args.csv.read_bytes()).hexdigest()}",
                 f"Benchmark run folder: {args.csv.resolve().parent.name}",
                 f"Plot output: {output}",
                 note, f"Fixed tile: {args.tile[0]}x{args.tile[1]}", f"Tile-comparison input: {size_label(args.fixed_size)}",
                 f"Figures: {len(writer.index)}; format: PNG",
                 "Missing and failed rows are never replaced by zero or interpolated.",
                 ("Bound placement flag covers before GPU, after GPU before CPU verification, and after verification; checks are snapshots."
                  if args.malloc_bound_only else "Host placement is only queried before and after each kernel, not continuously."),
                 "Times include the output stores to HBM; this is not isolated DRAM/NVLink latency.",
                 "Effective input GB/s = input_bytes / (kernel_ms * 1e6), recomputed from successful timings.",
                 "Effective input rate is not physical DRAM or NVLink bandwidth and does not count output bytes.",
                 "No automatic crossover estimate, error bars or averaging from this single-launch sweep.",
                 "checks.csv is an internal check, not a publication figure."]
        (checks / "plot_notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
        print(f"Saved {len(writer.index)} figures to {output.resolve()}")
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
