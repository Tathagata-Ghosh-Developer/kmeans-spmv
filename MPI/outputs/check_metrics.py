"""Recompute spmv_performance_metrics.csv from the raw run logs.

Checks two things:
  1. Every row of the CSV matches the mean over the 5 runs in
     logs/results_P<p>.txt (max/min/avg time, imbalance, speedup).
  2. The sampled outputs y[0], y[mid], y[end] are identical for every
     process count, i.e. the nnz-balanced partition does not change the
     product.

Usage: python check_metrics.py   (run from MPI/outputs)
"""
import csv
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = {"Max Time": r"Max Time:\s+([0-9.]+)\s+s",
          "Min Time": r"Min Time:\s+([0-9.]+)\s+s",
          "Avg Time": r"Avg Time:\s+([0-9.]+)\s+s",
          "Load Imbalance": r"Load Imbalance:\s+([0-9.]+)"}
SAMPLES = r"y\[(0|mid|end)\] = (\S+)"


def main():
    with open(os.path.join(HERE, "spmv_performance_metrics.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))

    failures = 0
    samples_seen = set()
    t1 = None
    for row in rows:
        p = int(row["Processors"])
        with open(os.path.join(HERE, "logs", f"results_P{p}.txt")) as fh:
            log = fh.read()
        recomputed = {}
        for name, pattern in FIELDS.items():
            vals = [float(v) for v in re.findall(pattern, log)]
            recomputed[name] = sum(vals) / len(vals)
        if p == 1:
            t1 = recomputed["Max Time"]
        recomputed["Speedup"] = t1 / recomputed["Max Time"]
        for name, value in recomputed.items():
            if not math.isclose(value, float(row[name]), rel_tol=1e-9, abs_tol=1e-12):
                print(f"MISMATCH P={p} {name}: csv={row[name]} logs={value}")
                failures += 1
        samples_seen.add(tuple(sorted(set(re.findall(SAMPLES, log)))))
        print(f"P={p:4d}  runs={len(re.findall(FIELDS['Max Time'], log))}  "
              f"max={recomputed['Max Time']:.6f} s  speedup={recomputed['Speedup']:.3f}x")

    if len(samples_seen) != 1:
        print(f"y samples differ across process counts: {samples_seen}")
        failures += 1
    else:
        print("y samples identical for all P:", dict((k, v) for k, v in next(iter(samples_seen))))

    print("OK" if failures == 0 else f"{failures} problem(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
