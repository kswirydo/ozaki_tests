#!/usr/bin/env python3
"""
plot_frobenius_3way.py
----------------------
Build 3 figures from frobenius_3way.log.

The log is a whitespace-aligned table:

    matrix                       shape        GB300vsGB200  GB200vsMI355X  GB300vsMI355X
    C_fp8_ozaki10_cond1e10.txt   4096x4096    0.000000e+00  0.000000e+00   0.000000e+00
    ...
    C_int8_ozaki16_cond1e9.txt   4096x4096    ...
    C_native_fp64_cond1e2.txt    4096x4096    ...

Each matrix name encodes:
  * a data type  : fp8 (Ozaki-II), int8 (Ozaki-II), or native_fp64
  * a condition  : cond1e<N>  ->  log10(cond) = N  (the x-axis of every figure)

The last three columns are the "3 diffs" (pairwise cross-GPU Frobenius-norm
differences); they become the 3 curves in each figure.

Figures produced (x-axis = log10(cond number), y-axis = the 3 diffs):
  * frobenius_3way_figemuFP8.png     <- rows whose type is fp8   (Ozaki-II)
  * frobenius_3way_figemuInt8.png    <- rows whose type is int8  (Ozaki-II)
  * frobenius_3way_fignativeFP64.png <- rows whose type is native_fp64

Note: fp8 has two moduli variants (ozaki10, ozaki12) and int8 has two
(ozaki12, ozaki16). To keep exactly 3 curves per figure, values are aggregated
(mean) across moduli variants per condition number. For the emulated cases the
variants are numerically identical (all 0), so this is lossless.

Usage:
  python3 plot_frobenius_3way.py [frobenius_3way.log]
"""
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COND_RE = re.compile(r"cond1e(\d+)", re.IGNORECASE)


def classify(name):
    """Return the data-type category for a matrix file name, or None."""
    if "native_fp64" in name:
        return "nativeFP64"
    if "fp8" in name:
        return "emuFP8"
    if "int8" in name:
        return "emuInt8"
    return None


def parse_log(path):
    """
    Parse the aligned table.

    Returns (diff_names, data) where
        diff_names = [col1, col2, col3]  (the 3 diff column headers)
        data[category][log10cond] = [ [d1,...], [d2,...], [d3,...] ]  (lists to average)
    """
    with open(path) as fh:
        lines = fh.readlines()

    # Header: first non-separator line containing "matrix".
    header = None
    start = 0
    for i, ln in enumerate(lines):
        if ln.strip().startswith("matrix"):
            header = ln.split()
            start = i + 1
            break
    if header is None:
        sys.exit("Could not find header line starting with 'matrix'.")
    diff_names = header[2:5]  # skip 'matrix' and 'shape'

    data = defaultdict(lambda: defaultdict(lambda: [[], [], []]))
    for ln in lines[start:]:
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("-"):
            continue
        parts = s.split()
        if len(parts) < 5:
            continue
        name = parts[0]
        cat = classify(name)
        m = COND_RE.search(name)
        if cat is None or m is None:
            continue
        log10cond = int(m.group(1))
        vals = [float(parts[-3]), float(parts[-2]), float(parts[-1])]
        for k in range(3):
            data[cat][log10cond][k].append(vals[k])
    return diff_names, data


def aggregate(cat_data):
    """cat_data[log10cond] = [[..],[..],[..]] -> sorted xs, [y1,y2,y3]."""
    xs = sorted(cat_data.keys())
    ys = [[], [], []]
    for x in xs:
        for k in range(3):
            series = cat_data[x][k]
            ys[k].append(sum(series) / len(series) if series else float("nan"))
    return xs, ys


STYLES = [
    dict(color="tab:blue",  marker="o", linestyle="-"),
    dict(color="tab:red",   marker="s", linestyle="--"),
    dict(color="tab:green", marker="^", linestyle="-."),
]


def display_label(name):
    """The GPU is B200 (not GB200); GB300 is left unchanged."""
    return re.sub(r"GB200", "B200", name)

TITLES = {
    "emuFP8":     "FP8 Ozaki-II emulation : cross-GPU Frobenius-norm differences",
    "emuInt8":    "INT8 Ozaki-II emulation : cross-GPU Frobenius-norm differences",
    "nativeFP64": "Native FP64 : cross-GPU Frobenius-norm differences",
}


def make_figure(cat, diff_names, cat_data, out_path):
    xs, ys = aggregate(cat_data)

    fig, ax = plt.subplots(figsize=(11, 7))
    for k in range(3):
        ax.plot(xs, ys[k], label=display_label(diff_names[k]), markersize=7,
                linewidth=1.9, **STYLES[k])

    # symlog handles both exact zeros and the enormous dynamic range (1e-10..1e19).
    ax.set_yscale("symlog", linthresh=1e-12)
    ax.set_xlabel("log10(condition number)   [cond = 10^x]",
                  fontsize=12, fontweight="bold")
    ax.set_ylabel("Frobenius-norm difference between GPU results  (symlog scale)",
                  fontsize=12, fontweight="bold")
    ax.set_title(TITLES[cat] + "\nmatrices 4096x4096", fontsize=13, fontweight="bold")

    if xs:
        ax.set_xticks(xs)
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(title="GPU pair compared", fontsize=11, title_fontsize=11,
              loc="best", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else "frobenius_3way.log"
    diff_names, data = parse_log(log_path)

    for cat in ("emuFP8", "emuInt8", "nativeFP64"):
        if cat not in data:
            print(f"WARNING: no rows found for category '{cat}', skipping.")
            continue
        make_figure(cat, diff_names, data[cat], f"frobenius_3way_fig{cat}.png")


if __name__ == "__main__":
    main()
