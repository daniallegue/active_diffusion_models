"""Per-round figure of the degenerate-bridge ablation (reports/reachability/degenerate_fixes.md).

Three panels on one round axis, each the mean over seeds of a per-round diagnostic that
`ABELoop` records in a cell's `rounds`:

    mu_cross_A    mu* mass on endpoints in A reached by crossing the band (what a path cost targets)
    band_known    share of the band the surrogate's hard rule already calls invalid
    A dirty       cumulative share of positives in A with a crossing anywhere in their ancestry

Usage:
    python -m adm.evaluation.plot_ablation --dir <sweep dir> --out reports/reachability/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

# Categorical slots 1-4 of the reference palette, in fixed order
ARMS = {
    "abe-hard": ("#2a78d6", "legacy hard"),
    "abe-hard+mass": ("#eb6834", "+ positive mass 0"),
    "abe-hard+all": ("#1baf7a", "+ all fixes"),
    "abe-oracle+all": ("#eda100", "oracle V + all fixes"),
}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def series(cells: list[dict], key: str, cumulative_share: bool = False) -> list[float]:
    per_seed = []
    for c in cells:
        rounds = c["rounds"]
        if cumulative_share:
            num = den = 0
            ys = []
            for r in rounds:
                num += r.get("query_A_dirty", 0)
                den += r.get("query_A", 0)
                ys.append(num / den if den else float("nan"))
        else:
            ys = [r.get(key, float("nan")) for r in rounds]
        per_seed.append(ys)

    n = min(len(s) for s in per_seed)
    out = []
    for i in range(n):
        xs = [s[i] for s in per_seed if s[i] == s[i]]
        out.append(sum(xs) / len(xs) if xs else float("nan"))
    return out


def main(args: argparse.Namespace):
    rows = [json.loads(p.read_text()) for p in sorted((args.dir / "results").glob("hook_*.json"))]
    panels = [
        ("mu_cross_A", False, "μ* mass in A reached by crossing"),
        ("band_known", False, "share of band the surrogate calls invalid"),
        ("A dirty", True, "positives in A with a crossing ancestor (cumulative)"),
    ]

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True, facecolor=SURFACE)
    for ax, (key, cum, title) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for arm, (color, label) in ARMS.items():
            cells = [r for r in rows if r["arm"] == arm and r.get("rounds")]
            if not cells:
                continue
            ys = series(cells, key, cum)
            ax.plot(range(1, len(ys) + 1), ys, color=color, lw=2, label=label)
        ax.set_title(title, color=INK, fontsize=9, loc="left")
        ax.set_xlabel("round")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_ylim(bottom=0)

    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "degenerate_rounds.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    print(f"wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/reachability/figures"))
    main(parser.parse_args())
