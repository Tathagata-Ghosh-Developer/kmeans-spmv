"""Summarise a K-means benchmark log written by run_experiments.sh.

Usage:
    python analysis.py ../results/kmeans_output_8094.txt [--plot-dir DIR]

The log contains the average sequential time and, for each thread count,
the average static- and dynamic-schedule times. Speedup is measured
against the sequential average (not against the slowest parallel run),
and efficiency is speedup / threads.
"""
import argparse
import re
import sys


def parse_log(path):
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()

    seq = re.search(r"Average Sequential Time:\s*([0-9.]+)\s*ms", text)
    if seq is None:
        sys.exit(f"{path}: no 'Average Sequential Time' line found")

    rows = []
    for threads, t_static, t_dynamic in re.findall(
        r"^(\d+),([0-9.]+),([0-9.]+)\s*$", text, flags=re.MULTILINE
    ):
        rows.append((int(threads), float(t_static), float(t_dynamic)))
    if not rows:
        sys.exit(f"{path}: no 'Threads,Static_avg_ms,Dynamic_avg_ms' rows found")

    repeat = re.search(r"averaged over (\d+) runs", text)
    return float(seq.group(1)), rows, int(repeat.group(1)) if repeat else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log", help="benchmark log (kmeans_output_<jobid>.txt)")
    ap.add_argument("--plot-dir", help="write speedup.png and efficiency.png here")
    args = ap.parse_args()

    t_seq, rows, repeat = parse_log(args.log)
    print(f"Sequential average: {t_seq:.4f} ms"
          + (f" (each value averaged over {repeat} runs)" if repeat else ""))
    print(f"{'threads':>7} {'static ms':>10} {'speedup':>8} {'eff':>6}"
          f" {'dynamic ms':>11} {'speedup':>8} {'eff':>6}")
    for threads, t_s, t_d in rows:
        s_s, s_d = t_seq / t_s, t_seq / t_d
        print(f"{threads:>7} {t_s:>10.4f} {s_s:>7.2f}x {s_s / threads:>6.2f}"
              f" {t_d:>11.4f} {s_d:>7.2f}x {s_d / threads:>6.2f}")

    if args.plot_dir:
        import os
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(args.plot_dir, exist_ok=True)
        threads = [r[0] for r in rows]
        sp_static = [t_seq / r[1] for r in rows]
        sp_dynamic = [t_seq / r[2] for r in rows]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(threads, threads, "k--", label="ideal")
        ax.plot(threads, sp_static, "o-", label="schedule(static)")
        ax.plot(threads, sp_dynamic, "s-", label="schedule(dynamic)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(threads, [str(t) for t in threads])
        ax.set_xlabel("OpenMP threads")
        ax.set_ylabel("speedup vs sequential")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.plot_dir, "kmeans_speedup.png"), dpi=150)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(threads, [s / t for s, t in zip(sp_static, threads)], "o-",
                label="schedule(static)")
        ax.plot(threads, [s / t for s, t in zip(sp_dynamic, threads)], "s-",
                label="schedule(dynamic)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(threads, [str(t) for t in threads])
        ax.set_xlabel("OpenMP threads")
        ax.set_ylabel("parallel efficiency")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.plot_dir, "kmeans_efficiency.png"), dpi=150)
        print(f"Plots written to {args.plot_dir}")


if __name__ == "__main__":
    main()
