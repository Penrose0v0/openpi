"""Aggregate sweep_angles.sh logs into a CSV and an SR-vs-azimuth plot.

Parses data/sweep/<MODEL_TAG>_<TASK_SUITE>_az<AZ>.log files (filename carries model/suite/
azimuth; the run logs "Total success rate: X"), writes a tidy CSV, and draws one line per
(model, suite) showing success rate vs camera azimuth — the core figure for the
multi-view-vs-single-view comparison.

Run (either env, needs matplotlib for the plot; CSV works without it):
    cd examples/libero
    python plot_sweep.py --sweep_dir data/sweep --out_csv sweep.csv --out_png sweep.png
"""
import argparse
import csv
import glob
import os
import re

RE_SR = re.compile(r"Total success rate:\s*([0-9.]+)")
# <model>_<suite>_az<az>.log ; suite is one of the known libero_* names.
RE_NAME = re.compile(r"^(?P<model>.+)_(?P<suite>libero_[a-z0-9]+)_az(?P<az>-?[0-9.]+)$")


def parse_log(path):
    """Return final success rate in [0,1], or None if the run didn't finish."""
    sr = None
    with open(path) as f:
        for line in f:
            m = RE_SR.search(line)
            if m:
                sr = float(m.group(1))
    return sr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_dir", default="data/sweep")
    p.add_argument("--out_csv", default="sweep.csv")
    p.add_argument("--out_png", default="sweep.png")
    p.add_argument("--per-suite", action="store_true",
                   help="also write one summary_<suite>.png/.csv per suite next to --out_png")
    args = p.parse_args()

    rows = []
    for log in sorted(glob.glob(os.path.join(args.sweep_dir, "*.log"))):
        stem = os.path.splitext(os.path.basename(log))[0]
        m = RE_NAME.match(stem)
        if not m:
            print(f"skip (unrecognized name): {log}")
            continue
        sr = parse_log(log)
        if sr is None:
            print(f"skip (no success rate / unfinished): {log}")
            continue
        rows.append({
            "model": m.group("model"),
            "suite": m.group("suite"),
            "azimuth": float(m.group("az")),
            "success_rate": sr,
        })

    if not rows:
        raise SystemExit(f"no parseable logs in {args.sweep_dir}")

    rows.sort(key=lambda r: (r["model"], r["suite"], r["azimuth"]))
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "suite", "azimuth", "success_rate"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out_csv} ({len(rows)} rows)")

    # Console summary: mean and worst-case SR across angles per (model, suite).
    groups = {}
    for r in rows:
        groups.setdefault((r["model"], r["suite"]), []).append(r)
    print(f"\n{'model':>12} {'suite':>16} {'mean_SR':>8} {'min_SR':>8} {'@45':>6}")
    for (model, suite), g in sorted(groups.items()):
        srs = [x["success_rate"] for x in g]
        at45 = next((x["success_rate"] for x in g if x["azimuth"] == 45), float("nan"))
        print(f"{model:>12} {suite:>16} {sum(srs) / len(srs):8.3f} {min(srs):8.3f} {at45:6.2f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed; CSV written, skipping plot.")
        return

    suites = sorted({r["suite"] for r in rows})
    fig, axes = plt.subplots(1, len(suites), figsize=(5 * len(suites), 4), squeeze=False)
    for ax, suite in zip(axes[0], suites):
        for model in sorted({r["model"] for r in rows if r["suite"] == suite}):
            g = sorted([r for r in rows if r["suite"] == suite and r["model"] == model],
                       key=lambda r: r["azimuth"])
            ax.plot([r["azimuth"] for r in g], [r["success_rate"] for r in g],
                    marker="o", label=model)
        ax.axvspan(0, 90, color="gray", alpha=0.08)  # training azimuth range
        ax.set_title(suite)
        ax.set_xlabel("camera azimuth delta (deg)")
        ax.set_ylabel("success rate")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=130)
    print(f"wrote {args.out_png}")

    if args.per_suite:
        base, ext = os.path.splitext(args.out_png)          # e.g. .../summary + .png
        csv_base = os.path.splitext(args.out_csv)[0]
        for suite in suites:
            srows = [r for r in rows if r["suite"] == suite]
            # per-suite csv
            with open(f"{csv_base}_{suite}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["model", "suite", "azimuth", "success_rate"])
                w.writeheader()
                w.writerows(srows)
            # per-suite png
            fig1, ax = plt.subplots(figsize=(5, 4))
            for model in sorted({r["model"] for r in srows}):
                g = sorted([r for r in srows if r["model"] == model], key=lambda r: r["azimuth"])
                ax.plot([r["azimuth"] for r in g], [r["success_rate"] for r in g],
                        marker="o", label=model)
            ax.axvspan(0, 90, color="gray", alpha=0.08)
            ax.set_title(suite)
            ax.set_xlabel("camera azimuth delta (deg)")
            ax.set_ylabel("success rate")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig1.tight_layout()
            fig1.savefig(f"{base}_{suite}{ext}", dpi=130)
            print(f"wrote {base}_{suite}{ext}")


if __name__ == "__main__":
    main()
