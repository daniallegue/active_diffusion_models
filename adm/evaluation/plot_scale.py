"""Figures of the islands/corridor scale-up (jobs/islands_scale.sh): every metric against the gap.

One figure per question, each a grid of families (rows) by metrics (columns), with the median over
seeds as a line and the interquartile range as a band:

    scale_references.png  ActFlow against ABE without a path cost, under each reference measure Q.
                     Does a heavy tail keep ABE ahead as the gap grows?
    scale_path_cost.png   ABE with and without the best path cost, under Gaussian and Levy Q. On the
                     islands no valid route exists, so the cost should cost A; on the corridor it
                     should only cost queries.

The metrics are the share of seeds that reached A, the queries to the first hit (over the seeds that
did), the final model's mass in A, and its mass in the forbidden band between S_0 and A. Every cell
is also written to `scale_summary.csv`, the table view of the same numbers.

`scale_paired.csv` holds the comparisons the conclusions rest on. Arms of one seed share a base
model, so they are compared seed by seed rather than as two independent samples: for each family
and gap, the wins of one arm over another, the median paired difference, and a two-sided Wilcoxon
signed-rank p-value. An unsolved run counts as one query past the budget, which is conservative for
whichever arm failed.

Usage:
    python -m adm.evaluation.plot_scale --dir <sweep dir> --out reports/reachability/figures
"""

import argparse
import csv
import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Categorical slots 1-5 of the reference palette, in fixed order. Slots 3-5 fall below 3:1 on the
# surface, so every series also has its own marker and the table is written alongside.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

FIGURES = {
    "references": {
        "actflow": "ActFlow",
        "abe+mass@gaussian": "ABE, Gaussian Q",
        "abe+mass@student": "ABE, Student-t Q (df 2)",
        "abe+mass@levy": "ABE, Lévy Q (α 1)",
        "abe+mass@poisson": "ABE, compound Poisson Q",
    },
    "path_cost": {
        "actflow": "ActFlow",
        "abe+mass@gaussian": "ABE, Gaussian Q",
        "abe-hard+all@gaussian": "ABE + hard cost, Gaussian Q",
        "abe+mass@levy": "ABE, Lévy Q",
        "abe-hard+all@levy": "ABE + hard cost, Lévy Q",
    },
}

METRICS = [
    ("solved", "reached A (share of seeds) ↑"),
    ("samples_to_solve", "queries to first hit in A ↓"),
    ("final_frac_in_target", "final model: mass in A ↑"),
    ("final_frac_shortcut", "final model: mass in forbidden band ↓"),
]

FAMILIES = {"islands": "disconnected islands", "corridor": "islands joined by a corridor"}


def quartiles(xs: list[float]) -> tuple[float, float, float]:
    if not xs:
        return float("nan"), float("nan"), float("nan")
    q1, q2, q3 = np.percentile(xs, [25, 50, 75])
    return float(q1), float(q2), float(q3)


def cell_stats(cells: list[dict]) -> dict[str, tuple[float, float, float]]:
    solved = [c for c in cells if c["solved"]]
    share = len(solved) / len(cells)
    return {
        "solved": (share, share, share),
        "samples_to_solve": quartiles([c["samples_to_solve"] for c in solved]),
        "final_frac_in_target": quartiles([c["final_frac_in_target"] for c in cells]),
        "final_frac_shortcut": quartiles([c["final_frac_shortcut"] for c in cells]),
    }


def write_table(rows: list[dict], path: Path):
    keys = sorted({(r["family"], r["length"], r["arm"]) for r in rows})
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "gap", "arm", "seeds", "solved", "queries_median", "queries_q1", "queries_q3",
                    "mass_A_median", "mass_A_q1", "mass_A_q3", "mass_band_median", "budget", "runtime_median_s"])
        for family, length, arm in keys:
            cells = [r for r in rows if (r["family"], r["length"], r["arm"]) == (family, length, arm)]
            s = cell_stats(cells)
            q, a = s["samples_to_solve"], s["final_frac_in_target"]
            w.writerow([family, length, arm, len(cells), f"{sum(c['solved'] for c in cells)}/{len(cells)}",
                        q[1], q[0], q[2], round(a[1], 4), round(a[0], 4), round(a[2], 4),
                        round(s["final_frac_shortcut"][1], 4), cells[0]["budget"],
                        round(median(c["runtime"] for c in cells), 1)])
    print(f"wrote {path}")


# (arm, baseline): every ABE arm against ActFlow, then each ablation against the arm it modifies
PAIRS = [
    *((f"abe+mass@{q}", "actflow") for q in ("gaussian", "student", "levy", "poisson")),
    *((f"abe-hard+all@{q}", "actflow") for q in ("gaussian", "levy")),
    *((f"abe+mass@{q}", "abe+mass@gaussian") for q in ("student", "levy", "poisson")),
    *((f"abe-hard+all@{q}", f"abe+mass@{q}") for q in ("gaussian", "levy")),
]


def paired_values(cells: list[dict], key: str) -> dict[tuple[float, int], float]:
    """Values keyed by (gap, seed), the unit two arms are paired on."""
    if key == "samples_to_solve":
        return {(c["length"], c["seed"]): c["samples_to_solve"] if c["solved"] else c["budget"] + 1 for c in cells}
    return {(c["length"], c["seed"]): c[key] for c in cells}


def write_paired(rows: list[dict], path: Path):
    from scipy.stats import wilcoxon

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "gap", "arm", "baseline", "metric", "n", "arm_better", "median_diff", "p_wilcoxon"])
        for family in FAMILIES:
            lengths = sorted({r["length"] for r in rows if r["family"] == family})
            # Each gap on its own, then every gap pooled: with 5 seeds a single gap cannot reach a
            # two-sided Wilcoxon p below 0.0625, so the pooled row is the one that can be conclusive
            for gaps in [[g] for g in lengths] + [lengths]:
                sub = [r for r in rows if r["family"] == family and r["length"] in gaps]
                for arm, base in PAIRS:
                    for key, lower_better in (("samples_to_solve", True), ("final_frac_in_target", False)):
                        a = paired_values([r for r in sub if r["arm"] == arm], key)
                        b = paired_values([r for r in sub if r["arm"] == base], key)
                        pairs = sorted(a.keys() & b.keys())
                        if not pairs:
                            continue

                        d = np.array([a[s] - b[s] for s in pairs], dtype=float)
                        better = int(((d < 0) if lower_better else (d > 0)).sum())
                        p = float(wilcoxon(d).pvalue) if np.any(d != 0) and len(d) > 1 else 1.0
                        w.writerow([family, gaps[0] if len(gaps) == 1 else "all", arm, base, key, len(pairs),
                                    better, round(float(np.median(d)), 4), round(p, 4)])
    print(f"wrote {path}")


def plot(rows: list[dict], arms: dict[str, str], path: Path):
    families = [f for f in FAMILIES if any(r["family"] == f for r in rows)]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(len(families), len(METRICS), figsize=(13, 3.1 * len(families)),
                             constrained_layout=True, facecolor=SURFACE, squeeze=False)

    for i, family in enumerate(families):
        sub = [r for r in rows if r["family"] == family]
        for j, (key, title) in enumerate(METRICS):
            ax = axes[i, j]
            ax.set_facecolor(SURFACE)
            for k, (arm, label) in enumerate(arms.items()):
                cells = [r for r in sub if r["arm"] == arm]
                if not cells:
                    continue

                gaps = sorted({c["length"] for c in cells})
                stats = [cell_stats([c for c in cells if c["length"] == g])[key] for g in gaps]
                lo, mid, hi = (np.array(v) for v in zip(*stats))
                ax.plot(gaps, mid, color=SLOTS[k], marker=MARKERS[k], ms=6, lw=2, label=label,
                        mec=SURFACE, mew=1.5, zorder=3)
                if key != "solved":
                    ax.fill_between(gaps, lo, hi, color=SLOTS[k], alpha=0.14, lw=0, zorder=2)

            if key == "samples_to_solve":
                # Queries come in rounds of 32, so powers of two read as whole rounds
                ax.set_yscale("log", base=2)
                ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            elif key == "solved":
                ax.set_ylim(-0.04, 1.04)
            else:
                ax.set_ylim(bottom=0)

            ax.set_title(title if i == 0 else "", color=INK, fontsize=9, loc="left")
            ax.set_xlabel("gap ℓ" if i == len(families) - 1 else "")
            if j == 0:
                ax.set_ylabel(FAMILIES[family], color=INK)
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8.5,
               labelcolor=INK, bbox_to_anchor=(0.5, -0.06))
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main(args: argparse.Namespace):
    rows = [json.loads(p.read_text()) for p in sorted((args.dir / "results").glob("*.json"))]
    rows = [r for r in rows if r["family"] in FAMILIES and r.get("length") is not None]
    if not rows:
        raise SystemExit(f"no islands or corridor results with a length in {args.dir / 'results'}")

    args.out.mkdir(parents=True, exist_ok=True)
    write_table(rows, args.out / "scale_summary.csv")
    write_paired(rows, args.out / "scale_paired.csv")
    for name, arms in FIGURES.items():
        plot(rows, arms, args.out / f"scale_{name}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/reachability/figures"))
    main(parser.parse_args())
