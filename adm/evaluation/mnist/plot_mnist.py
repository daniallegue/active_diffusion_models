import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from diffusiongym import DDTensor  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

from adm.setups.mnist_latent import MNISTLatentProblemSetup  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
VALID, REJECTED = "#008300", "#e34948"

# Fixed colour per arm, so an arm keeps its colour in every panel it appears in
ARMS = {
    "actflow": ("ActFlow, DPS 30", 0),
    "actflow-nodps": ("ActFlow, no DPS", 3),
    "actflow@dps100": ("ActFlow, DPS 100", 4),
    "actflow@dps300": ("ActFlow, DPS 300", 6),
    "abe+mass@gaussian": ("ABE, Gaussian Q", 1),
    "abe+mass@student": ("ABE, Student-t Q", 2),
    "abe+mass@levy": ("ABE, Lévy Q", 7),
}
PANELS = {
    "ActFlow: DPS ablation": ["actflow-nodps", "actflow", "actflow@dps100", "actflow@dps300"],
    "ActFlow against ABE": ["actflow", "abe+mass@gaussian", "abe+mass@student", "abe+mass@levy"],
}
PROCESS_ARMS = ["actflow-nodps", "actflow", "actflow@dps300", "abe+mass@gaussian", "abe+mass@levy"]
SNAPSHOTS = [0, 320, 960, 1600, 3200]


def load(dir: Path) -> list[dict]:
    rows = [json.loads(p.read_text()) for p in sorted((dir / "results").glob("*.json"))]
    for r in rows:
        r["trace"] = torch.load(dir / "runs" / f"{r['arm']}_seed{r['seed']}" / "trace.pt")
    return rows


def coverage(row: dict, island_class: dict[int, int], new: set[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cumulative islands hit and new classes found after each round, against queries."""
    seen: set[int] = set()
    queries, islands, classes = [0], [0], [0]
    for t in row["trace"]:
        seen |= {c for c in t["comp"][t["valid"]].tolist() if c in island_class}
        queries.append(int(t["queried"]))
        islands.append(len(seen))
        classes.append(len({island_class[c] for c in seen} & new))
    return np.array(queries), np.array(islands), np.array(classes)


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_coverage(rows, island_class, new, path):
    budget = max(r["budget"] for r in rows)
    grid = np.linspace(0, budget, 101)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), constrained_layout=True, facecolor=SURFACE, sharex=True)
    for j, (title, arms) in enumerate(PANELS.items()):
        for i, (key, ylabel, top) in enumerate([(1, "islands hit", len(island_class)),
                                                (2, "new digit classes found", len(new))]):
            ax = axes[i, j]
            style(ax)
            for arm in arms:
                cells = [r for r in rows if r["arm"] == arm]
                if not cells:
                    continue

                label, slot = ARMS[arm]
                curves = []
                for r in cells:
                    c = coverage(r, island_class, new)
                    # A step function: the count after the last round that had finished by q
                    curves.append(c[key][np.searchsorted(c[0], grid, side="right") - 1])
                lo, mid, hi = np.percentile(np.array(curves), [25, 50, 75], axis=0)
                ax.plot(grid, mid, color=SLOTS[slot], lw=2, label=label, drawstyle="steps-post")
                ax.plot(grid[::20], mid[::20], ls="none", marker=MARKERS[slot], ms=6, color=SLOTS[slot],
                        mec=SURFACE, mew=1.5)
                ax.fill_between(grid, lo, hi, color=SLOTS[slot], alpha=0.14, lw=0, step="post")

            ax.set_ylim(-0.3, top + 0.3)
            ax.set_ylabel(f"{ylabel} (of {top})", color=INK)
            if i == 0:
                ax.set_title(title, color=INK, loc="left", fontsize=9.5)
                ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
            else:
                ax.set_xlabel("verifier queries", color=INK)

    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def island_hits(rows, islands) -> list[dict]:
    out = []
    for arm in [a for a in ARMS if any(r["arm"] == a for r in rows)]:
        cells = [r for r in rows if r["arm"] == arm]
        for isl in islands:
            hits = [h["first_hit"] for r in cells for h in r["islands"] if h["component"] == isl["component"]]
            done = [h for h in hits if h is not None]
            # Unhit seeds count as one query past the budget, which is conservative for that arm
            censored = [h if h is not None else cells[0]["budget"] + 1 for h in hits]
            out.append({"arm": arm, "component": isl["component"], "class": isl["majority"],
                        "gap": round(isl["gap"], 4), "data_share": round(isl["data_share"], 4),
                        "seeds": len(cells), "seeds_hit": len(done),
                        "median_first_hit": float(np.median(censored)) if len(done) * 2 > len(cells) else None})
    return out


def plot_gap_hit(rows, table, path):
    budget = max(r["budget"] for r in rows)
    never = budget * 1.6
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True, facecolor=SURFACE, sharey=True)
    for ax, (title, arms) in zip(axes, PANELS.items()):
        style(ax)
        for k, arm in enumerate(arms):
            pts = sorted((r["gap"], r["median_first_hit"]) for r in table if r["arm"] == arm)
            if not pts:
                continue

            label, slot = ARMS[arm]
            gaps = np.array([g for g, _ in pts])
            ys = np.array([never if h is None else h for _, h in pts])
            # A small horizontal offset per arm, so arms that hit the same island do not overplot
            xs = gaps * (1 + 0.03 * (k - 1.5))
            ax.plot(xs, ys, color=SLOTS[slot], lw=1.2, alpha=0.5)
            ax.plot(xs, ys, ls="none", marker=MARKERS[slot], ms=7, color=SLOTS[slot], mec=SURFACE, mew=1.5,
                    label=label)

        ax.axhline(budget, color=MUTED, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("island gap to S₀ in Z (log)", color=INK)
        ax.set_title(title, color=INK, loc="left", fontsize=9.5)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, -0.18),
                  ncol=2)
        ax.set_yticks([64, 256, 1024, budget, never], ["64", "256", "1024", f"{budget} budget", "never"])
    axes[0].set_ylabel("queries to first hit (median over seeds)", color=INK)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def snapshot(dir: Path, row: dict, q: int) -> torch.Tensor:
    """The model's samples after `q` queries: the eval taken at that round, or the final samples."""
    run = dir / "runs" / f"{row['arm']}_seed{row['seed']}"
    if q >= row["budget"]:
        return torch.load(run / "final_samples.pt")

    return torch.load(run / "eval" / f"{q // row['per_round']:04d}" / "samples.pt")


def island_map(problem: MNISTLatentProblemSetup, hit: set[int]) -> np.ndarray:
    rgb = np.empty(problem.lab.shape + (3,))
    rgb[:] = to_rgb("#ecebe7")
    rgb[np.isin(problem.lab, [r["component"] for r in problem.components if r["seed"]])] = to_rgb("#9ec5f4")
    rgb[np.isin(problem.lab, [r["component"] for r in problem.islands])] = to_rgb("#f8cdb9")
    rgb[np.isin(problem.lab, list(hit))] = to_rgb("#eb6834")
    return rgb


def plot_process(dir, rows, problem, seed, path):
    arms = [a for a in PROCESS_ARMS if any(r["arm"] == a and r["seed"] == seed for r in rows)]
    fig, axes = plt.subplots(len(arms), len(SNAPSHOTS), figsize=(2.6 * len(SNAPSHOTS), 2.5 * len(arms)),
                             constrained_layout=True, facecolor=SURFACE, squeeze=False)
    island_ids = {r["component"] for r in problem.islands}
    for i, arm in enumerate(arms):
        row = next(r for r in rows if r["arm"] == arm and r["seed"] == seed)
        prev = 0
        for j, q in enumerate(SNAPSHOTS):
            ax = axes[i, j]
            done = [t for t in row["trace"] if int(t["queried"]) <= q]
            window = [t for t in done if int(t["queried"]) > prev]
            hit = {c for t in done for c in t["comp"][t["valid"]].tolist() if c in island_ids}
            ax.imshow(island_map(problem, hit), extent=problem.extent, origin="lower", aspect="auto",
                      interpolation="nearest")

            z = snapshot(dir, row, q).numpy()
            ax.scatter(z[:, 0], z[:, 1], s=0.6, color=INK, alpha=0.35, lw=0, rasterized=True)
            if window:
                zq = torch.cat([t["z"] for t in window]).numpy()
                ok = torch.cat([t["valid"] for t in window]).numpy()
                ax.scatter(zq[ok, 0], zq[ok, 1], s=7, color=VALID, lw=0)
                ax.scatter(zq[~ok, 0], zq[~ok, 1], s=9, color=REJECTED, marker="x", lw=0.8)

            ax.set_xlim(problem.extent[0], problem.extent[1])
            ax.set_ylim(problem.extent[2], problem.extent[3])
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_color(MUTED)
            ax.set_title(f"{q} queries · {len(hit)} islands", color=INK, fontsize=8.5, loc="left")
            if j == 0:
                ax.set_ylabel(ARMS[arm][0], color=INK, fontsize=9)
            prev = q

    fig.suptitle(f"Sampling process, seed {seed}: S₀ blue, islands orange (dark once hit), model samples black, "
                 f"queries since the last column green (valid) and red (rejected)", color=INK, fontsize=9.5,
                 x=0.01, ha="left")
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


@torch.no_grad()
def plot_digits(dir, rows, problem, seed, path, n: int = 12):
    arms = [a for a in PROCESS_ARMS if any(r["arm"] == a and r["seed"] == seed for r in rows)]
    fig, axes = plt.subplots(len(arms), len(SNAPSHOTS), figsize=(2.6 * len(SNAPSHOTS), 1.3 * len(arms) + 0.3),
                             constrained_layout=True, facecolor=SURFACE, squeeze=False)
    gen = torch.Generator().manual_seed(0)
    for i, arm in enumerate(arms):
        row = next(r for r in rows if r["arm"] == arm and r["seed"] == seed)
        for j, q in enumerate(SNAPSHOTS):
            z = snapshot(dir, row, q)
            pick = z[torch.randperm(len(z), generator=gen)[:n]]
            imgs = problem.vae.decode(pick.to(problem.device)).cpu().numpy()[:, 0]
            sheet = imgs.reshape(2, n // 2, 28, 28).transpose(0, 2, 1, 3).reshape(56, 28 * n // 2)
            new_mass = problem.compute_metrics(DDTensor(z), {})["frac_in_target"]
            ax = axes[i, j]
            ax.imshow(1 - sheet, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_color(MUTED)
            ax.set_title(f"{q} queries · {new_mass:.0%} on islands", color=INK, fontsize=8.5, loc="left")
            if j == 0:
                ax.set_ylabel(ARMS[arm][0], color=INK, fontsize=9)

    fig.suptitle(f"Random decoded samples of the model, seed {seed}", color=INK, fontsize=9.5, x=0.01, ha="left")
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/mnist_islands/figures"))
    parser.add_argument("--process_seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = load(args.dir)
    first = rows[0]
    run_args = yaml.safe_load((args.dir / "runs" / f"{first['arm']}_seed{first['seed']}" / "args.yaml").read_text())
    problem = MNISTLatentProblemSetup(run_args, device=torch.device("cpu"))
    island_class = {r["component"]: r["majority"] for r in problem.islands}
    new = set(range(10)) - set(problem.seed_classes)

    plt.rcParams.update({"font.size": 9})
    plot_coverage(rows, island_class, new, args.out / "mnist_coverage.png")
    table = island_hits(rows, sorted(problem.islands, key=lambda r: r["gap"]))
    plot_gap_hit(rows, table, args.out / "mnist_gap_hit.png")
    plot_process(args.dir, rows, problem, args.process_seed, args.out / "mnist_process.png")
    plot_digits(args.dir, rows, problem, args.process_seed, args.out / "mnist_digits.png")

    path = args.out / "mnist_islands.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
