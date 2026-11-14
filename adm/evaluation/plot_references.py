import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from adm.abe import ABEConfig, sample_reference  # noqa: E402

INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
# The colours of these arms in plot_mnist, so the figures read together
REFERENCES = {
    "gaussian": ("Gaussian", "#eb6834"),
    "student": ("Student-t, 2 df", "#1baf7a"),
    "levy": ("Lévy, α = 1", "#e34948"),
}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=13)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("reports/mnist_islands/figures"))
    parser.add_argument("--n", type=int, default=2000, help="steps drawn in each cloud")
    parser.add_argument("--view", type=float, default=8.0, help="half-width of the clouds, in units of s")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # The parameters of the sweeps' references; no max_jump, to show the law itself
    cfg = ABEConfig(reference_df=2.0, reference_alpha=1.0, max_jump=float("inf"))
    device = torch.device("cpu")
    torch.manual_seed(0)

    plt.rcParams.update({"font.size": 12})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.9), constrained_layout=True, facecolor=SURFACE)

    for j, (ref, (name, colour)) in enumerate(REFERENCES.items()):
        z = sample_reference(args.n, 2, 1.0, cfg, device, reference=ref).numpy()
        beyond = (np.abs(z) > args.view).any(axis=1).mean()

        ax = axes[j]
        style(ax)
        ax.scatter(z[:, 0], z[:, 1], s=6, color=colour, alpha=0.45, lw=0)
        ax.plot(0, 0, marker="+", color=INK, ms=14, mew=2)
        ax.set_xlim(-args.view, args.view)
        ax.set_ylim(-args.view, args.view)
        ax.set_aspect("equal")
        ticks = np.linspace(-args.view, args.view, 5)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_title(name, color=INK, fontsize=22, loc="left")
        ax.set_xlabel("$z_1$", color=INK, fontsize=20)
        if j == 0:
            ax.set_ylabel("$z_2$", color=INK, fontsize=20)
        ax.text(0.03, 0.03, f"{beyond:.0%} of steps land\noutside this view", transform=ax.transAxes,
                color=MUTED, fontsize=16, va="bottom",
                bbox={"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.85, "pad": 2})

    path = args.out / "references.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
