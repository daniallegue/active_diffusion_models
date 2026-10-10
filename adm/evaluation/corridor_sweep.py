"""Sweep ActFlow over $(w, \\ell)$ corridors and measure the cost of crossing them.

For every (width, length, seed) cell, ActFlow is run on the `corridor` problem setup until a valid
sample lands in the target island A, or until the budget of `num_iters * samples_per_iter` samples
runs out. The cost T is the number of exploration samples drawn until then. Afterwards, the cells are
aggregated into a contour map of T over (l, w) and the log-linear model

    log T = c + a log l + b log w

is fitted to the solved cells. The corridor bound predicts T ~ l / w^3, i.e., a = 1 and b = -3.

Usage:
    python -m adm.corridor_sweep gp --dir <dir> --widths 0.1 0.2 0.4 --lengths 0.5 1 2 --seeds 0 1 2
    python -m adm.corridor_sweep gp --dir <dir> ... --grid_index $SLURM_ARRAY_TASK_ID  # one cell
    python -m adm.corridor_sweep gp --dir <dir> --plot_only
"""

from typing import Any
import argparse
import itertools
import json
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
from diffusiongym import DummyReward
from pathlib import Path

from ..explore import build_loop, add_global_args
from ..setups.corridor import CorridorProblemSetup
from ..uncertainty import uncertainty_estimators


def grid(args: argparse.Namespace) -> list[tuple[float, float, int]]:
    return list(itertools.product(args.widths, args.lengths, args.seeds))


def cell_name(w: float, l: float, seed: int) -> str:
    return f"w{w:g}_l{l:g}_s{seed}"


def run_cell(args: argparse.Namespace, w: float, l: float, seed: int):
    out = args.dir / "results" / f"{cell_name(w, l, seed)}.json"
    if out.exists():
        print(f"skipping {out.name}, already done")
        return

    cell_args = argparse.Namespace(**{
        **vars(args),
        "problem_setup": "corridor",
        "dir": args.dir / "runs" / cell_name(w, l, seed),
        "seed": seed,
        "corridor_width": w,
        "corridor_length": l,
        "corridor_base_ckpt": str(args.dir / "base_models" / f"seed{seed}.pt"),
        "stop_when_solved": True,
    })

    t0 = time.perf_counter()
    loop = build_loop(cell_args, reward=DummyReward(), mean_weight=0)
    summary = loop.explore_loop(loop.config.num_iters, loop.config.samples_per_iter)
    runtime = time.perf_counter() - t0

    # Where did the final model's support get to?
    with torch.no_grad():
        sample = loop.env.batch_sample(args.final_eval_samples, args.eval_batch_size).sample
    metrics = loop.problem.compute_metrics(sample, {})

    result = {
        "width": w,
        "length": l,
        "seed": seed,
        "budget": loop.config.num_iters * loop.config.samples_per_iter,
        "runtime": runtime,
        **summary,
        **{f"final_{k}": v for k, v in metrics.items()},
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result))


def load_results(dir: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted((dir / "results").glob("*.json"))]


def fit_power_law(lengths: np.ndarray, widths: np.ndarray, costs: np.ndarray) -> dict[str, float]:
    """OLS fit of log T = c + a log l + b log w, plus the fit with exponents fixed to (1, -3)."""
    y = np.log(costs)
    X = np.stack([np.ones_like(y), np.log(lengths), np.log(widths)], axis=1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = max(len(y) - X.shape[1], 1)
    cov = (resid @ resid / dof) * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    ss_tot = ((y - y.mean()) ** 2).sum()

    y_theory = np.log(lengths) - 3 * np.log(widths)
    c_theory = (y - y_theory).mean()
    resid_theory = y - (c_theory + y_theory)

    return {
        "n": len(y),
        "c": coef[0], "a": coef[1], "b": coef[2],
        "c_se": se[0], "a_se": se[1], "b_se": se[2],
        "r2": 1 - (resid @ resid) / ss_tot if ss_tot > 0 else float("nan"),
        "c_theory": c_theory,
        "r2_theory": 1 - (resid_theory @ resid_theory) / ss_tot if ss_tot > 0 else float("nan"),
    }


def plot(args: argparse.Namespace):
    results = load_results(args.dir)
    if len(results) == 0:
        raise RuntimeError(f"no results found in {args.dir / 'results'}")

    widths = np.array(sorted({r["width"] for r in results}))
    lengths = np.array(sorted({r["length"] for r in results}))

    # Aggregate seeds: unsolved runs count as +inf (right-censored), so a
    # cell is only censored when at least half of its seeds did not reach A.
    median = np.full((len(widths), len(lengths)), np.nan)
    solved_frac = np.zeros_like(median)
    for i, w in enumerate(widths):
        for j, l in enumerate(lengths):
            cell = [r for r in results if r["width"] == w and r["length"] == l]
            if len(cell) == 0:
                continue
            costs = [r["samples_to_solve"] if r["solved"] else np.inf for r in cell]
            median[i, j] = np.median(costs)
            solved_frac[i, j] = np.mean([r["solved"] for r in cell])

    solved = [r for r in results if r["solved"]]
    fit = None
    if len(solved) >= 4:
        fit = fit_power_law(
            np.array([r["length"] for r in solved]),
            np.array([r["width"] for r in solved]),
            np.array([r["samples_to_solve"] for r in solved], dtype=float),
        )
        with open(args.dir / "fit.json", "w") as f:
            json.dump(fit, f, indent=2)
        print(json.dumps(fit, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    plot_geometry(axes[0], widths, lengths, args)
    plot_contour(fig, axes[1], widths, lengths, median, solved_frac, fit)
    plot_fit(axes[2], solved, fit)

    fig.savefig(args.dir / "corridor_cost.png", dpi=200)
    fig.savefig(args.dir / "corridor_cost.pdf")
    print(f"saved {args.dir / 'corridor_cost.png'}")


def plot_geometry(ax, widths: np.ndarray, lengths: np.ndarray, args: argparse.Namespace):
    """Draw Omega_star for the extreme (w, l) cells, stacked vertically."""
    s = args.corridor_island_size
    corners = [(widths.min(), lengths.min()), (widths.max(), lengths.min()), (widths.min(), lengths.max())]
    gap = 0.4
    for k, (w, l) in enumerate(corners):
        setup = CorridorProblemSetup.__new__(CorridorProblemSetup)
        setup.width, setup.length, setup.island_size, setup.x0 = w, l, s, args.corridor_x0
        dy = -k * (s + gap)
        for box, color in ((setup.source_box, "#2a78b5"), (setup.corridor_box, "#888888"), (setup.target_box, "#d0762b")):
            xmin, xmax, ymin, ymax = box
            ax.add_patch(Rectangle((xmin, ymin + dy), xmax - xmin, ymax - ymin, facecolor=color, alpha=0.35, edgecolor=color, lw=1))
        ax.text(setup.x0 - s - 0.1, dy, f"w={w:g}\nl={l:g}", ha="right", va="center", fontsize=8)

    ax.text(args.corridor_x0 - s / 2, s / 2 + 0.05, r"$S_0$", ha="center", va="bottom", fontsize=9)
    ax.text(args.corridor_x0 + lengths.min() + s / 2, s / 2 + 0.05, r"$A$", ha="center", va="bottom", fontsize=9)
    ax.set_xlim(args.corridor_x0 - s - 0.9, args.corridor_x0 + lengths.max() + s + 0.1)
    ax.set_ylim(-(len(corners) - 1) * (s + gap) - s / 2 - 0.1, s / 2 + 0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(r"Valid region $\Omega_\star = S_0 \cup C \cup A$", fontsize=10)


def plot_contour(fig, ax, widths, lengths, median, solved_frac, fit):
    L, W = np.meshgrid(lengths, widths)
    logT = np.ma.masked_invalid(np.log10(median))

    if logT.count() >= 4 and len(widths) >= 2 and len(lengths) >= 2:
        cs = ax.contourf(L, W, logT, levels=12, cmap="Blues")
        fig.colorbar(cs, ax=ax, label=r"$\log_{10} T$ (median samples to reach $A$)")

    # Theory iso-lines T = e^c * l / w^3, with only the constant fitted
    if fit is not None:
        ll, ww = np.meshgrid(np.linspace(lengths.min(), lengths.max(), 100), np.geomspace(widths.min(), widths.max(), 100))
        theory = np.log10(np.exp(fit["c_theory"]) * ll / ww ** 3)
        levels = np.arange(np.floor(np.nanmin(theory)), np.ceil(np.nanmax(theory)) + 0.5, 0.5)
        ct = ax.contour(ll, ww, theory, levels=levels, colors="#d0762b", linestyles="--", linewidths=1)
        ax.clabel(ct, fmt=lambda v: f"{v:.1f}", fontsize=7)
        ax.plot([], [], color="#d0762b", ls="--", lw=1, label=r"theory $T \propto \ell / w^3$")

    # Cells: filled = median solved, cross = censored (most seeds never reached A within budget)
    censored = ~np.isfinite(median)
    ax.scatter(L[~censored], W[~censored], s=14, color="#1b3a5c", label="reached $A$")
    ax.scatter(L[censored], W[censored], s=36, marker="x", color="#b23b3b", label="not reached (censored)")
    for i in range(len(widths)):
        for j in range(len(lengths)):
            if 0 < solved_frac[i, j] < 1:
                ax.annotate(f"{solved_frac[i, j]:.0%}", (L[i, j], W[i, j]), textcoords="offset points", xytext=(4, 4), fontsize=7, color="#444444")

    ax.set_yscale("log")
    ax.set_yticks(widths)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel(r"corridor length $\ell$")
    ax.set_ylabel(r"corridor width $w$")
    ax.set_title("ActFlow cost to cross the corridor", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")


def plot_fit(ax, solved: list[dict[str, Any]], fit):
    ax.set_title(r"Fit $\log T = c + a \log \ell + b \log w$", fontsize=10)
    if fit is None:
        ax.text(0.5, 0.5, "not enough solved runs to fit", ha="center", va="center", transform=ax.transAxes)
        return

    l = np.array([r["length"] for r in solved])
    w = np.array([r["width"] for r in solved])
    T = np.array([r["samples_to_solve"] for r in solved], dtype=float)
    pred = np.exp(fit["c"]) * l ** fit["a"] * w ** fit["b"]

    sc = ax.scatter(pred, T, c=np.log10(w), cmap="Blues", vmin=np.log10(w.min()) - 0.5, edgecolor="#1b3a5c", lw=0.5, s=24)
    lo, hi = min(pred.min(), T.min()) / 1.5, max(pred.max(), T.max()) * 1.5
    ax.plot([lo, hi], [lo, hi], color="#888888", lw=1, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("fitted T")
    ax.set_ylabel("observed T (samples)")
    plt.colorbar(sc, ax=ax, label=r"$\log_{10} w$")

    text = (
        f"$a = {fit['a']:.2f} \\pm {fit['a_se']:.2f}$  (theory 1)\n"
        f"$b = {fit['b']:.2f} \\pm {fit['b_se']:.2f}$  (theory $-3$)\n"
        f"$R^2 = {fit['r2']:.2f}$,  $R^2_{{(1,-3)}} = {fit['r2_theory']:.2f}$\n"
        f"$n = {fit['n']}$ solved runs"
    )
    ax.text(0.03, 0.97, text, transform=ax.transAxes, va="top", fontsize=8, bbox=dict(facecolor="white", edgecolor="#cccccc"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="uncertainty_estimator", required=True)

    for u_name, u_cls in uncertainty_estimators.items():
        sub = subparsers.add_parser(u_name)
        add_global_args(sub)
        CorridorProblemSetup.add_args(sub)
        u_cls.add_args(sub)

        sub.add_argument("--no_uncertainty", action="store_true")
        sub.add_argument("--no_verifier", action="store_true")

        sub.add_argument("--widths", type=float, nargs="+", default=[0.1, 0.15, 0.2, 0.3, 0.45, 0.6])
        sub.add_argument("--lengths", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        sub.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
        sub.add_argument("--grid_index", type=int, default=None, help="Only run this cell of the grid (e.g. SLURM array id)")
        sub.add_argument("--final_eval_samples", type=int, default=2048)
        sub.add_argument("--plot_only", action="store_true")
        sub.add_argument("--no_plot", action="store_true")

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    if not args.plot_only:
        cells = grid(args)
        if args.grid_index is not None:
            cells = [cells[args.grid_index]]

        for w, l, seed in cells:
            run_cell(args, w, l, seed)

    if not args.no_plot and args.grid_index is None:
        plot(args)
