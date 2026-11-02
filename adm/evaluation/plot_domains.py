"""Draws the reachability domains, straight from `adm.setups.domains`.

The figure is generated from the same objects the sweep runs on, so it cannot drift from the
geometry it documents: the valid region is `Domain.boxes`, the shaded band is `Domain.crossed`
evaluated on a grid, and the dashed path is `Domain.route`. Nothing here is hand-drawn.

    python -m adm.evaluation.plot_domains --out reports/reachability/figures

`domains` itself needs only torch, but importing it goes through `adm/setups/__init__.py`,
which loads every setup, so this still needs the full environment until that package is made lazy.
"""

from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from ..setups import domains  # noqa: E402


# Slots 1 and 3 of the reference categorical palette, plus a reserved critical red for the
# forbidden band. Every element is also directly labelled, so identity is never colour alone.
SOURCE = "#2a78d6"
TARGET = "#1baf7a"
FORBIDDEN = "#e34948"
VALID_FILL = "#e8e7e2"
VALID_EDGE = "#9a9890"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
SURFACE = "#fcfcfb"


def draw(ax, domain: domains.Domain, title: str, extent=None, grid: int = 420):
    # A shared extent across panels, so the two geometries are directly comparable and equal aspect
    # does not give them different sizes.
    xmin, xmax, ymin, ymax = extent if extent is not None else domain.view_extent

    # The forbidden band, computed rather than drawn: exactly `Domain.crossed`, the mass that can
    # only have arrived by crossing, which is what `frac_shortcut` counts.
    xs = torch.linspace(xmin, xmax, grid)
    ys = torch.linspace(ymin, ymax, grid)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    pts = torch.stack([gx.flatten(), gy.flatten()], dim=1)
    crossed = domain.crossed(pts).reshape(grid, grid).T.float()

    ax.set_facecolor(SURFACE)
    ax.imshow(
        crossed,
        extent=(xmin, xmax, ymin, ymax),
        origin="lower",
        cmap=matplotlib.colors.ListedColormap([(0, 0, 0, 0), matplotlib.colors.to_rgba(FORBIDDEN, 0.17)]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="equal",
        zorder=1,
    )

    for box in domain.boxes:
        bx, bX, by, bY = box
        if box == domain.source:
            face, edge, z = matplotlib.colors.to_rgba(SOURCE, 0.22), SOURCE, 3
        elif box == domain.target:
            face, edge, z = matplotlib.colors.to_rgba(TARGET, 0.22), TARGET, 3
        else:
            face, edge, z = VALID_FILL, VALID_EDGE, 2

        ax.add_patch(Rectangle((bx, by), bX - bx, bY - by, facecolor=face, edgecolor=edge,
                               lw=1.2, zorder=z))

    # The valid route, as arclength is measured along it
    rx = [p[0] for p in domain.route]
    ry = [p[1] for p in domain.route]
    ax.plot(rx, ry, color=INK, lw=2.0, ls="--", dashes=(4, 3), zorder=5,
            solid_capstyle="round", label="valid route")

    # The straight line an ambient jump would take
    a, b = domain._centre(domain.source), domain._centre(domain.target)
    ax.annotate("", xy=b, xytext=a, zorder=4,
                arrowprops=dict(arrowstyle="-|>", color=FORBIDDEN, lw=1.8, shrinkA=0, shrinkB=0))

    ax.text(*a, "  $S_0$", color=SOURCE, fontsize=11, fontweight="bold", va="center", zorder=6)
    ax.text(*b, "  $A$", color=TARGET, fontsize=11, fontweight="bold", va="center", zorder=6)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(VALID_EDGE)
        spine.set_linewidth(0.8)

    ax.set_title(title, fontsize=11, color=INK, pad=8)
    steps = domain.required_steps()
    ax.text(
        0.5, 0.02,
        f"detour {domain.detour_ratio:.2f}×   ·   wall {domain.wall_thickness():.2f}   ·   "
        f"{steps} step{'s' if steps != 1 else ''}",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color=INK_SOFT, zorder=7,
    )


def main(out: Path, width: float, length: float, gap: float):
    corridor = domains.corridor(width, length)
    hook = domains.hook(width, length, gap)

    # One extent for both panels: same scale, same box, so the shapes can be compared by eye
    exts = [d.view_extent for d in (corridor, hook)]
    extent = (min(e[0] for e in exts), max(e[1] for e in exts),
              min(e[2] for e in exts), max(e[3] for e in exts))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    draw(axes[0], corridor, "corridor — the control", extent)
    draw(axes[1], hook, "hook — where the claim is testable", extent)

    handles = [
        plt.Line2D([], [], color=INK, lw=2, ls="--", label="valid route (geodesic)"),
        plt.Line2D([], [], color=FORBIDDEN, lw=1.8, label="straight line $S_0 \\to A$"),
        plt.Rectangle((0, 0), 1, 1, facecolor=matplotlib.colors.to_rgba(FORBIDDEN, 0.17),
                      edgecolor="none", label="forbidden band (counted by frac_shortcut)"),
        plt.Rectangle((0, 0), 1, 1, facecolor=VALID_FILL, edgecolor=VALID_EDGE,
                      label="valid region"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9,
               labelcolor=INK_SOFT, bbox_to_anchor=(0.5, -0.03))

    out.mkdir(parents=True, exist_ok=True)
    path = out / "geometry.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")
    for name, d in (("corridor", corridor), ("hook", hook)):
        print(f"  {name:<9} detour {d.detour_ratio:.2f}  wall {d.wall_thickness():.3f}  steps {d.required_steps()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("reports/reachability/figures"))
    ap.add_argument("--width", type=float, default=0.2)
    ap.add_argument("--length", type=float, default=1.5)
    ap.add_argument("--gap", type=float, default=0.3)
    a = ap.parse_args()
    main(a.out, a.width, a.length, a.gap)
