#!/usr/bin/env python3
"""
plot_scaling_sweep3.py
----------------------
Graph the two CSV files produced by run_scaling_sweep3.sh (INT8 + FP8):

  * scaling_sweep3_norms_*.csv  -> Frobenius norms of C1 .. C6
  * scaling_sweep3_diffs_*.csv  -> relative norm differences (7 curves)

The x-axis is the exponent interval size (the sweep's 4th parameter). Each
figure draws one explicitly-labeled curve per CSV column and is saved next to
the input CSV (only the extension changes).

Usage:
  python3 plot_scaling_sweep3.py NORMS_CSV DIFFS_CSV
  python3 plot_scaling_sweep3.py NORMS_CSV DIFFS_CSV --format pdf --dpi 600
  # or, with no CSV arguments, it auto-detects the most recent matching CSVs.
"""
import argparse
import csv
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt

# Human-readable names for the raw CSV column headers.
COLUMN_LABELS = {
    "||C1||_F": "C1 = A*B (FP64)",
    "||C2||_F": "C2 = A_scal*B_scal (FP64)",
    "||C3||_F": "C3 = A_scal*B_scal (INT8 Ozaki II)",
    "||C4||_F": "C4 = A*B (INT8 Ozaki II)",
    "||C5||_F": "C5 = A_scal*B_scal (FP8 Ozaki II)",
    "||C6||_F": "C6 = A*B (FP8 Ozaki II)",
    "|C1-C2|/|C1|": "FP64 scaled vs FP64 ref",
    "|C1-C3|/|C1|": "INT8 scaled vs FP64 ref",
    "|C1-C4|/|C1|": "INT8 unscaled vs FP64 ref",
    "|C1-C5|/|C1|": "FP8 scaled vs FP64 ref",
    "|C1-C6|/|C1|": "FP8 unscaled vs FP64 ref",
    "|C3-C4|/|C4|": "INT8 scaled vs INT8 unscaled",
    "|C5-C6|/|C6|": "FP8 scaled vs FP8 unscaled",
}


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
    """Pull N / seed / moduli / GPU out of the filename for a descriptive title."""
    base = os.path.basename(path)
    parts = []
    n = re.search(r"_N(\d+)_", base)
    if n:
        parts.append(f"N = K = M = {n.group(1)}")
    seed = re.search(r"_seed(\d+)_", base)
    if seed:
        parts.append(f"seed = {seed.group(1)}")
    int8m = re.search(r"_int8m(\d+)_", base)
    if int8m:
        parts.append(f"INT8 moduli = {int8m.group(1)}")
    fp8m = re.search(r"_fp8m(\d+)_", base)
    if fp8m:
        parts.append(f"FP8 moduli = {fp8m.group(1)}")
    gpu = re.search(r"_fp8m\d+_(.+)\.csv$", base)
    if gpu:
        parts.append(f"GPU = {gpu.group(1).replace('_', ' ')}")
    return ", ".join(parts)


# Distinct styles so all curves stay unambiguous, including in grayscale.
STYLES = [
    dict(color="tab:blue", marker="o", linestyle="-"),
    dict(color="tab:green", marker="s", linestyle="--"),
    dict(color="tab:red", marker="^", linestyle="-."),
    dict(color="tab:purple", marker="D", linestyle=":"),
    dict(color="tab:orange", marker="v", linestyle="-"),
    dict(color="tab:brown", marker="P", linestyle="--"),
    dict(color="tab:cyan", marker="X", linestyle="-."),
]


def plot_file(path, title_prefix, ylabel, yscale, out_path, dpi):
    _, x, cols = read_csv(path)

    fig, ax = plt.subplots(figsize=(11, 7))
    for idx, (name, ys) in enumerate(cols.items()):
        style = STYLES[idx % len(STYLES)]
        label = COLUMN_LABELS.get(name, name)
        ax.plot(x, ys, label=f"{name}  ({label})", markersize=6, linewidth=1.8, **style)

    ax.set_yscale(yscale, **({"linthresh": 1e-17} if yscale == "symlog" else {}))
    ax.set_xlabel("Exponent interval size  (r_k drawn from [-range, +range])",
                  fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")

    ax.set_title(f"{title_prefix}\n{meta_from_name(path)}", fontsize=13, fontweight="bold")

    ax.set_xticks(x)
    ax.tick_params(axis="x", labelrotation=0, labelsize=9)
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(title="Curve", fontsize=9, title_fontsize=10, loc="best", framealpha=0.9)

    fig.tight_layout()
    save_kwargs = {"dpi": dpi}
    if os.path.splitext(out_path)[1].lower() in (".jpg", ".jpeg"):
        save_kwargs["pil_kwargs"] = {"quality": 95, "optimize": True}
    fig.savefig(out_path, **save_kwargs)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def autodetect():
    norms = sorted(glob.glob("scaling_sweep3_norms_*.csv"))
    diffs = sorted(glob.glob("scaling_sweep3_diffs_*.csv"))
    if not norms or not diffs:
        sys.exit("No scaling_sweep3_*.csv files found; pass them explicitly.")
    return norms[-1], diffs[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("norms_csv", nargs="?", help="scaling_sweep3_norms_*.csv")
    parser.add_argument("diffs_csv", nargs="?", help="scaling_sweep3_diffs_*.csv")
    parser.add_argument("--format", choices=("png", "pdf", "jpg", "jpeg"), default="png",
                        help="Output image format (default: png)")
    parser.add_argument("--dpi", type=int, default=150, help="Raster resolution (default: 150)")
    args = parser.parse_args()

    if args.norms_csv and args.diffs_csv:
        norms_csv, diffs_csv = args.norms_csv, args.diffs_csv
    elif not args.norms_csv and not args.diffs_csv:
        norms_csv, diffs_csv = autodetect()
        print(f"Auto-detected:\n  {norms_csv}\n  {diffs_csv}")
    else:
        sys.exit("Pass both CSVs or neither.")

    ext = "." + args.format

    plot_file(
        norms_csv,
        title_prefix="Frobenius norms vs exponent interval size (INT8 + FP8 Ozaki II)",
        ylabel="Frobenius norm  ||C||_F   (log scale)",
        yscale="log",
        out_path=os.path.splitext(norms_csv)[0] + ext,
        dpi=args.dpi,
    )
    plot_file(
        diffs_csv,
        title_prefix="Relative Frobenius-norm differences vs exponent interval size (INT8 + FP8 Ozaki II)",
        ylabel="Relative difference  (symlog scale)",
        yscale="symlog",
        out_path=os.path.splitext(diffs_csv)[0] + ext,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
