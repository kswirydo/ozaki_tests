#!/usr/bin/env python3
"""Compare matrices (matrix-by-matrix) between two result directories.

For every matrix file that exists in both directories (matched by filename),
load both as 2D arrays and report the Frobenius norm of their difference:

    ||A - B||_F = sqrt(sum((A - B)^2))

Files are assumed to be dense matrices with comma-separated values, one row
per line. The Frobenius norm is computed in a streaming, row-by-row fashion so
that the (potentially very large) matrices never need to be held in memory in
their entirety.
"""

import argparse
import csv
import os
import sys

import numpy as np


DEFAULT_DIR_A = "c_results"
DEFAULT_DIR_B = "c_results_b300"


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
    """Load a comma-separated matrix, returning (flat_values, rows, cols).

    Parsing is done with numpy for speed. We track the per-row column count so
    that ragged matrices can be reported as shape mismatches.
    """
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


def frobenius_diff(path_a, path_b):
    """Frobenius norm of (A - B); also returns shape info.

    Returns a tuple: (frob_norm, rows, cols, mismatch_message_or_None).
    """
    a, ra, ca = load_matrix(path_a)
    b, rb, cb = load_matrix(path_b)
    if (ra, ca) != (rb, cb):
        return (None, ra, ca,
                f"shape differs (A is {ra}x{ca}, B is {rb}x{cb})")
    frob = float(np.linalg.norm(a - b))
    return (frob, ra, ca, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dir_a", nargs="?", default=DEFAULT_DIR_A,
                        help=f"first results directory (default: {DEFAULT_DIR_A})")
    parser.add_argument("dir_b", nargs="?", default=DEFAULT_DIR_B,
                        help=f"second results directory (default: {DEFAULT_DIR_B})")
    parser.add_argument("-o", "--output", default=None,
                        help="optional CSV file to write results to")
    args = parser.parse_args()

    for d in (args.dir_a, args.dir_b):
        if not os.path.isdir(d):
            sys.exit(f"error: not a directory: {d}")

    files_a = matrix_files(args.dir_a)
    files_b = matrix_files(args.dir_b)

    common = sorted(files_a & files_b)
    only_a = sorted(files_a - files_b)
    only_b = sorted(files_b - files_a)

    if only_a:
        print(f"# {len(only_a)} file(s) only in {args.dir_a}: "
              f"{', '.join(only_a)}", file=sys.stderr)
    if only_b:
        print(f"# {len(only_b)} file(s) only in {args.dir_b}: "
              f"{', '.join(only_b)}", file=sys.stderr)

    rows_out = []
    print(f"{'matrix':45s} {'frobenius_diff':>16s}  shape")
    print("-" * 80)
    for name in common:
        path_a = os.path.join(args.dir_a, name)
        path_b = os.path.join(args.dir_b, name)
        try:
            frob, r, c, err = frobenius_diff(path_a, path_b)
        except ValueError as e:
            print(f"{name:45s} {'PARSE ERROR':>16s}  ({e})")
            rows_out.append((name, "", "", str(e)))
            continue
        if err is not None:
            print(f"{name:45s} {'SHAPE MISMATCH':>16s}  ({err})")
            rows_out.append((name, "", f"{r}x{c}", err))
            continue
        print(f"{name:45s} {frob:16.6e}  {r}x{c}")
        rows_out.append((name, f"{frob:.6e}", f"{r}x{c}", ""))

    if args.output:
        with open(args.output, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["matrix", "frobenius_diff", "shape", "note"])
            w.writerows(rows_out)
        print(f"\n# wrote results to {args.output}")


if __name__ == "__main__":
    main()
