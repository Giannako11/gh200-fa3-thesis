"""Simple empirical sparsity histograms from existing sampled activations.

One view summarizes each generation; another counts encountered token/layer/
tensor vectors. Neither assumes a Gaussian distribution or reruns the model.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from protocol import atomic_json, complete_result


def histogram(values):
    """Equal-width empirical bins; report an interval, not an exact modal point."""
    values = np.asarray(values, dtype=float)
    low, high = float(values.min()), float(values.max())
    if low == high:
        width = max(abs(low) * .02, 1e-6)
        edges = np.array([max(0, low - width/2), high + width/2])
    else:
        q25, q75 = np.quantile(values, [.25, .75])
        width = 2 * (q75-q25) / len(values)**(1/3)
        bins = int(np.clip(np.ceil((high-low)/width), 12, 40)) if width > 0 else 20
        edges = np.linspace(low, np.nextafter(high, np.inf), bins+1)
    counts, _ = np.histogram(values, bins=edges)
    return edges, counts / len(values), counts


def collect(root):
    manifest = json.loads((root / "manifest.json").read_text())
    metrics = manifest["protocol"]["metrics"]
    zero_index = metrics.index("zero_count")
    # One exact atom at zero, then 0.005-percentage-point intervals for the
    # positive tail. Values above 0.3% stay in an explicit overflow category.
    positive_edges = np.linspace(0, .3, 61)
    encountered = np.zeros(62, dtype=float)
    rows, count = [], 0
    for case in manifest["cases"]:
        folder = root / "cases" / case["id"]
        result = complete_result(folder, manifest["identity"], case)
        if result is None:
            continue
        with np.load(folder / "observations.npz", allow_pickle=False) as saved:
            zeros = saved["metrics"][:, zero_index].astype(np.float64)
            widths = saved["metadata"][:, 5].astype(np.float64)
        percentages = 100 * zeros / widths
        if np.any(zeros < 0) or np.any(zeros > widths) or not np.isfinite(percentages).all():
            raise ValueError(f"Invalid zero counts: {folder}")
        vector_counts = np.concatenate(([np.count_nonzero(zeros == 0)],
                          np.histogram(percentages[(zeros > 0) & (percentages <= .3)], bins=positive_edges)[0],
                          [np.count_nonzero(percentages > .3)]))
        if vector_counts.sum() != len(zeros):
            raise ValueError("Encountered-vector histogram lost observations")
        encountered += vector_counts / len(zeros)  # Equal mass per generation.
        rows.append({"id": case["id"], "prompt_index": case["prompt_index"],
                     "input_tokens": case["input_tokens"], "sampled_vectors": len(zeros),
                     "sampled_elements": int(widths.sum()), "zero_elements": int(zeros.sum()),
                     "sampled_sparsity_pct": float(100 * zeros.sum() / widths.sum())})
        count += 1
        if count % 300 == 0:
            print(f"Frequency histograms: read {count} generations", flush=True)
    if not rows:
        raise ValueError("No PASS generations available")
    return rows, encountered / count, positive_edges


def plot(root, output=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root).resolve()
    output = Path(output).resolve() if output else root / "plots"
    output.mkdir(parents=True, exist_ok=True)
    rows, encountered, positive_edges = collect(root)
    values = np.asarray([r["sampled_sparsity_pct"] for r in rows])
    edges, probability, counts = histogram(values)
    centers = (edges[:-1] + edges[1:]) / 2
    modes = np.flatnonzero(counts == counts.max())
    mode = int(modes[0])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})

    fig, ax = plt.subplots(figsize=(11, 6.7))
    fig.subplots_adjust(left=.12, right=.97, top=.78, bottom=.23)
    fig.text(.12, .94, "Πόση sparsity συναντάμε ανά inference run;", fontsize=19, weight="bold")
    fig.text(.12, .875, f"{len(rows):,} generations · ακριβή μηδενικά στα δειγματοληπτημένα activations", color="#475569")
    colours = ["#a8cbd4"] * len(counts)
    for i in modes:
        colours[int(i)] = "#157986"
    ax.bar(centers, 100 * probability, width=np.diff(edges)*.92, color=colours, edgecolor="white", linewidth=.7)
    # This polygon follows the observed frequencies; no normal curve is fitted.
    ax.plot(centers, 100*probability, color="#0b4b58", linewidth=1.6, marker="o", markersize=3)
    ax.axvline(values.mean(), color="#475569", linestyle="--", linewidth=1.2,
               label=f"Μέση τιμή: {values.mean():.5f}%")
    ax.set_xlabel("Μέση sparsity των δειγμάτων του run (%)", labelpad=12)
    ax.set_ylabel("Ποσοστό generations ανά διάστημα (%)", labelpad=10)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    ax.text(.02, .96, f"Συχνότερο διάστημα\n{edges[mode]:.5f}% – {edges[mode+1]:.5f}%\n{100*probability[mode]:.1f}% των generations",
            transform=ax.transAxes, va="top", fontsize=11,
            bbox={"boxstyle": "round,pad=.6", "facecolor": "#f0f7f8", "edgecolor": "#d8e7eb"})
    fig.text(.12, .105, "Κάθε generation δίνει μία τιμή: συνολικά μηδενικά / συνολικά μετρημένα στοιχεία.", fontsize=10)
    fig.text(.12, .065, "Αφορά τα δείγματα των 5 tensor types και των επιλεγμένων θέσεων· όχι όλα τα activations του μοντέλου.", fontsize=9, color="#64748b")
    fig.text(.12, .03, "Πραγματικές συχνότητες, χωρίς υπόθεση κανονικής κατανομής. Τα 7 μήκη των ίδιων prompts είναι συσχετισμένα.", fontsize=9, color="#64748b")
    first = output / "sparsity_frequency_per_generation.png"
    fig.savefig(first, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6.7))
    fig.subplots_adjust(left=.12, right=.97, top=.78, bottom=.24)
    fig.text(.12, .94, "Τι sparsity έχουν τα activations που συναντάμε;", fontsize=18, weight="bold")
    fig.text(.12, .875, "Ακριβή μηδενικά ανά δειγματοληπτημένο token / layer / tensor type", color="#475569")
    width = positive_edges[1]-positive_edges[0]
    centers = (positive_edges[:-1]+positive_edges[1:])/2
    ax.bar(centers, 100*encountered[1:-1], width=width*.9, color="#8ab8c7")
    # Place the point mass separately at x=0; the other bars exclude zero.
    ax.bar([0], [100*encountered[0]], width=width*.65, color="#157986")
    ax.set(xlim=(-.007, .305), xlabel="Sparsity της παρατήρησης (%)",
           ylabel="Ποσοστό παρατηρήσεων (%) — στάθμιση ανά generation")
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    vector_mode = int(encountered.argmax())
    if vector_mode == 0:
        mode_text = f"Συχνότερη τιμή: 0%\n{100*encountered[0]:.2f}% των σταθμισμένων παρατηρήσεων\nδεν έχουν κανένα ακριβές μηδενικό."
    elif vector_mode == len(encountered)-1:
        mode_text = f"Συχνότερη κατηγορία: πάνω από 0,3%\n{100*encountered[-1]:.2f}% των σταθμισμένων παρατηρήσεων"
    else:
        mode_text = (f"Συχνότερο διάστημα: {positive_edges[vector_mode-1]:.3f}% – {positive_edges[vector_mode]:.3f}%\n"
                     f"{100*encountered[vector_mode]:.2f}% των σταθμισμένων παρατηρήσεων")
    ax.text(.36, .93, mode_text, transform=ax.transAxes,
            va="top", fontsize=12, bbox={"boxstyle": "round,pad=.7", "facecolor": "#f0f7f8", "edgecolor": "#d8e7eb"})
    fig.text(.12, .12, "Η μπάρα στο 0% είναι ξεχωριστή σημειακή μάζα. Τα επόμενα bins έχουν πλάτος 0,005 ποσοστιαίες μονάδες.", fontsize=9)
    fig.text(.12, .075, "Κάθε generation έχει ίσο συνολικό βάρος· μέσα του οι δειγματοληπτημένες feature vectors έχουν ίσο βάρος.", fontsize=9, color="#64748b")
    fig.text(.12, .03, f"Πάνω από 0,3%: {100*encountered[-1]:.5f}% των σταθμισμένων παρατηρήσεων. Όχι ανεξάρτητα δείγματα ή Gaussian fit.", fontsize=9, color="#64748b")
    second = output / "sparsity_frequency_encountered_vectors.png"
    fig.savefig(second, dpi=180)
    plt.close(fig)

    with (output / "sparsity_frequency_per_generation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"generations": len(rows), "mean_sampled_sparsity_pct": float(values.mean()),
               "minimum_sampled_sparsity_pct": float(values.min()), "maximum_sampled_sparsity_pct": float(values.max()),
               "modal_intervals": [{"low_pct": float(edges[i]), "high_pct": float(edges[i+1]),
                                    "probability": float(probability[i]), "generations": int(counts[i])} for i in modes],
               "encountered_vectors_probability_zero_sparsity": float(encountered[0]),
               "generation_bin_rule": "equal-width Freedman-Diaconis bins, clipped to 12–40 bins; the modal interval depends on the chosen binning",
               "generation_definition": "element-weighted exact-zero fraction across stored activation samples, one value per PASS generation",
               "vector_definition": "one sampled token/layer/tensor vector; equal weight per generation and per sampled vector within it",
               "scope": "selected positions of 5 tensor types, not exhaustive inference sparsity; empirical frequencies, no Gaussian assumption"}
    atomic_json(output / "sparsity_frequency_summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(first)
    print(second)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plot(args.run_directory, args.output)
