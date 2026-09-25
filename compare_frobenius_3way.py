#!/usr/bin/env python3
"""Three-way matrix comparison across GB300, GB200 and MI355X result sets.

For every matrix file present in the directories, load each as a 2D array and
report the Frobenius norm of the pairwise differences:

    ||A - B||_F = sqrt(sum((A - B)^2))

The three pairings are:

    * GB300 vs GB200
    * GB200 vs MI355X
    * GB300 vs MI355X

All results are written to a single CSV file (one row per matrix).

Default directory mapping:
    GB300  -> c_results_b300
    GB200  -> c_results_b200
    MI355X -> c_results

Matrices are comma-separated values, one row per line.
"""

import argparse
import csv
import os
import sys

import numpy as np


DEFAULT_DIR_GB300 = "c_results_b300"
DEFAULT_DIR_GB200 = "c_results_b200"
DEFAULT_DIR_MI355X = "c_results"


def matrix_files(directory):
    """Return the set of candidate matrix filenames in a directory."""
    names = set()
    for name in os.listdir(directory):
        if not name.endswith(".txt"):
            continue
        if not os.path.isfile(os.path.join(directory, name)):
            continue
        names.add(name)
    return names


def load_matrix(path):
    """Load a comma-separated matrix, returning (flat_values, rows, cols)."""
    rows = 0
    cols = None
    flat = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = np.fromstring(line, sep=",")
            if cols is None:
                cols = vals.size
            elif vals.size != cols:
                raise ValueError(
                    f"row {rows + 1} has {vals.size} columns, expected {cols}")
            flat.append(vals)
            rows += 1
    if not flat:
        return np.empty(0), 0, 0
    return np.concatenate(flat), rows, cols


def frobenius_diff(a, sa, b, sb):
    """Frobenius norm of (A - B). Returns (frob_or_None, note)."""
    if sa != sb:
        return None, f"shape differs ({sa[0]}x{sa[1]} vs {sb[0]}x{sb[1]})"
    return float(np.linalg.norm(a - b)), ""


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gb300", default=DEFAULT_DIR_GB300,
                        help=f"GB300 directory (default: {DEFAULT_DIR_GB300})")
    parser.add_argument("--gb200", default=DEFAULT_DIR_GB200,
                        help=f"GB200 directory (default: {DEFAULT_DIR_GB200})")
    parser.add_argument("--mi355x", default=DEFAULT_DIR_MI355X,
                        help=f"MI355X directory (default: {DEFAULT_DIR_MI355X})")
    parser.add_argument("-o", "--output", default="frobenius_3way.csv",
                        help="CSV file to write results to "
                             "(default: frobenius_3way.csv)")
    args = parser.parse_args()

    dirs = {"GB300": args.gb300, "GB200": args.gb200, "MI355X": args.mi355x}
    for label, d in dirs.items():
        if not os.path.isdir(d):
            sys.exit(f"error: {label} is not a directory: {d}")

    files = {label: matrix_files(d) for label, d in dirs.items()}

    # Matrices present in all three directories.
    common = sorted(files["GB300"] & files["GB200"] & files["MI355X"])

    for label, names in files.items():
        extra = sorted(names - set(common))
        if extra:
            print(f"# {len(extra)} file(s) not common (in {label}): "
                  f"{', '.join(extra)}", file=sys.stderr)

    header = ["matrix", "shape",
              "frob_GB300_vs_GB200",
              "frob_GB200_vs_MI355X",
              "frob_GB300_vs_MI355X",
              "note"]

    rows_out = []
    print(f"{'matrix':40s} {'shape':>12s} "
          f"{'GB300vsGB200':>16s} {'GB200vsMI355X':>16s} {'GB300vsMI355X':>16s}")
    print("-" * 110)

    for name in common:
        try:
            gb300, r3, c3 = load_matrix(os.path.join(args.gb300, name))
            gb200, r2, c2 = load_matrix(os.path.join(args.gb200, name))
            mi, rm, cm = load_matrix(os.path.join(args.mi355x, name))
        except ValueError as e:
            print(f"{name:40s} {'PARSE ERROR':>12s}  ({e})")
            rows_out.append([name, "", "", "", "", str(e)])
            continue

        s3, s2, sm = (r3, c3), (r2, c2), (rm, cm)
        f_3_2, n1 = frobenius_diff(gb300, s3, gb200, s2)
        f_2_m, n2 = frobenius_diff(gb200, s2, mi, sm)
        f_3_m, n3 = frobenius_diff(gb300, s3, mi, sm)

        note = "; ".join(n for n in (n1, n2, n3) if n)
        shape = f"{r3}x{c3}" if s3 == s2 == sm else \
            f"GB300={r3}x{c3},GB200={r2}x{c2},MI355X={rm}x{cm}"

        def fmt(v):
            return f"{v:.6e}" if v is not None else ""

        print(f"{name:40s} {shape:>12s} "
              f"{fmt(f_3_2):>16s} {fmt(f_2_m):>16s} {fmt(f_3_m):>16s}")
        rows_out.append([name, shape, fmt(f_3_2), fmt(f_2_m), fmt(f_3_m), note])

    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows_out)
    print(f"\n# wrote {len(rows_out)} rows to {args.output}")


if __name__ == "__main__":
    main()
