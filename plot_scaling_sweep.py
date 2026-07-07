#!/usr/bin/env python3
"""
plot_scaling_sweep.py
---------------------
Graph the two CSV files produced by run_scaling_sweep.sh:

  * scaling_sweep_norms_*.csv  -> Frobenius norms of C1, C2, C3, C4
  * scaling_sweep_diffs_*.csv  -> relative norm differences (4 curves)

The x-axis is the exponent interval size (the sweep's 4th parameter). Each
figure draws 4 explicitly-labeled curves and is saved as a PNG whose name
matches the input CSV (only the extension changes to .png).

Usage:
  python3 plot_scaling_sweep.py NORMS_CSV DIFFS_CSV
  # or, with no arguments, it auto-detects the most recent matching CSVs.
"""
import csv
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt


def read_csv(path):
    """Return (header_list, x_values, {col_name: [y_values]})."""
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [row for row in reader if row and row[0].strip() != ""]

    x = [float(r[0]) for r in rows]
    cols = {}
    for j in range(1, len(header)):
        cols[header[j]] = [float(r[j]) for r in rows]
    return header, x, cols


def meta_from_name(path):
    """Pull N / seed / GPU out of the filename for a descriptive title."""
    base = os.path.basename(path)
    n = re.search(r"_N(\d+)_", base)
    seed = re.search(r"_seed(\d+)_", base)
    gpu = re.search(r"_seed\d+_(.+)\.csv$", base)
    parts = []
    if n:
        parts.append(f"N = K = M = {n.group(1)}")
    if seed:
        parts.append(f"seed = {seed.group(1)}")
    if gpu:
        parts.append(f"GPU = {gpu.group(1).replace('_', ' ')}")
    return ", ".join(parts)


# Distinct styles so all 4 curves are unambiguous even in grayscale.
STYLES = [
    dict(color="tab:blue",   marker="o", linestyle="-"),
    dict(color="tab:green",  marker="s", linestyle="--"),
    dict(color="tab:red",    marker="^", linestyle="-."),
    dict(color="tab:purple", marker="D", linestyle=":"),
]


def plot_file(path, title_prefix, ylabel, yscale, out_path):
    header, x, cols = read_csv(path)

    fig, ax = plt.subplots(figsize=(11, 7))
    for idx, (name, ys) in enumerate(cols.items()):
        style = STYLES[idx % len(STYLES)]
        ax.plot(x, ys, label=name, markersize=6, linewidth=1.8, **style)

    ax.set_yscale(yscale, **({"linthresh": 1e-17} if yscale == "symlog" else {}))
    ax.set_xlabel("Exponent interval size  (r_k drawn from [-range, +range])",
                  fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")

    meta = meta_from_name(path)
    ax.set_title(f"{title_prefix}\n{meta}", fontsize=13, fontweight="bold")

    ax.set_xticks(x)
    ax.tick_params(axis="x", labelrotation=0, labelsize=9)
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(title="Curve", fontsize=11, title_fontsize=11,
              loc="best", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def autodetect():
    norms = sorted(glob.glob("scaling_sweep_norms_*.csv"))
    diffs = sorted(glob.glob("scaling_sweep_diffs_*.csv"))
    if not norms or not diffs:
        sys.exit("No scaling_sweep_*.csv files found; pass them explicitly.")
    return norms[-1], diffs[-1]


def main():
    if len(sys.argv) == 3:
        norms_csv, diffs_csv = sys.argv[1], sys.argv[2]
    elif len(sys.argv) == 1:
        norms_csv, diffs_csv = autodetect()
        print(f"Auto-detected:\n  {norms_csv}\n  {diffs_csv}")
    else:
        sys.exit("Usage: python3 plot_scaling_sweep.py [NORMS_CSV DIFFS_CSV]")

    plot_file(
        norms_csv,
        title_prefix="Frobenius norms vs exponent interval size",
        ylabel="Frobenius norm  ||C||_F   (log scale)",
        yscale="log",
        out_path=os.path.splitext(norms_csv)[0] + ".png",
    )
    plot_file(
        diffs_csv,
        title_prefix="Relative Frobenius-norm differences vs exponent interval size",
        ylabel="Relative difference  (symlog scale)",
        yscale="symlog",
        out_path=os.path.splitext(diffs_csv)[0] + ".png",
    )


if __name__ == "__main__":
    main()
