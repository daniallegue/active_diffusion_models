import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import ndimage  # noqa: E402

from adm.setups.mnist_latent import (  # noqa: E402
    VAE, batched, classify_grid, get_classifiers, get_vae, label_islands, latent_extent, load_mnist,
)

SEED_FILL, ISLAND_FILL, SPECK_FILL, INVALID_FILL = "#2a78d6", "#eb6834", "#a3a29c", "#ecebe7"
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


# -- figures -------------------------------------------------------------------------------------

def style(ax, extent):
    ax.set_facecolor(SURFACE)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("$z_1$", color=INK)
    ax.set_ylabel("$z_2$", color=INK)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ax.spines.values():
        side.set_color(MUTED)


def label_point(mask: np.ndarray) -> tuple[int, int]:
    """The cell deepest inside the component, so the label sits on it however it is shaped."""
    d = ndimage.distance_transform_edt(mask)
    iy, ix = np.unravel_index(d.argmax(), d.shape)
    return iy, ix


def plot_islands(args, lab, rows, cls, conf, tau, extent, path):
    fills = {"seed": to_rgb(SEED_FILL), "island": to_rgb(ISLAND_FILL), "speck": to_rgb(SPECK_FILL)}
    rgb = np.empty(lab.shape + (3,))
    rgb[:] = to_rgb(INVALID_FILL)
    for r in rows:
        kind = "seed" if r["seed"] else "speck" if r["speck"] else "island"
        rgb[lab == r["component"]] = fills[kind]

    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), constrained_layout=True, facecolor=SURFACE)
    xmin, xmax, ymin, ymax = extent
    xs, ys = np.linspace(xmin, xmax, args.grid_res), np.linspace(ymin, ymax, args.grid_res)

    ax = axes[0]
    ax.imshow(rgb, extent=extent, origin="lower", aspect="auto", interpolation="nearest")
    ax.contour(xs, ys, lab > 0, levels=[0.5], colors=INK, linewidths=0.6)
    for r in rows:
        if r["speck"]:
            continue
        iy, ix = label_point(lab == r["component"])
        text = str(r["majority"]) + ("+" if r["touching"] else "")
        if not r["seed"]:
            text += f"\ng={r['gap']:.2f}"
        ax.text(xs[ix], ys[iy], text, color="#ffffff", ha="center", va="center", fontsize=7.5,
                fontweight="bold", linespacing=0.9)
    style(ax, extent)
    n_islands = sum(not r["seed"] and not r["speck"] for r in rows)
    ax.set_title(f"Ω* at τ={tau:g}: {n_islands} islands outside S₀ (seeds {args.seed_classes})",
                 color=INK, loc="left", fontsize=9.5)
    ax.legend(handles=[Patch(color=SEED_FILL, label="S₀ (holds seed codes)"),
                       Patch(color=ISLAND_FILL, label="island: digit, gap g to S₀; + = mixed classes"),
                       Patch(color=SPECK_FILL, label=f"speck (< {args.min_cells} cells)"),
                       Patch(color=INVALID_FILL, label="rejected")],
              loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=8, labelcolor=INK)

    ax = axes[1]
    shown = np.where(cls >= 0, conf, 0.0)
    im = ax.imshow(shown, extent=extent, origin="lower", aspect="auto", cmap=BLUES, vmin=0, vmax=1,
                   interpolation="nearest")
    ax.contour(xs, ys, shown, levels=[tau], colors=INK, linewidths=0.8)
    style(ax, extent)
    ax.set_title("ensemble confidence (0 where members disagree), τ contour", color=INK, loc="left", fontsize=9.5)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_edgecolor(MUTED)

    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


@torch.no_grad()
def plot_manifold(vae: VAE, extent, device, path, n: int = 24):
    xmin, xmax, ymin, ymax = extent
    xs, ys = torch.linspace(xmin, xmax, n), torch.linspace(ymax, ymin, n)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    imgs = vae.decode(torch.stack([X.flatten(), Y.flatten()], 1).to(device)).cpu().numpy()
    sheet = imgs.reshape(n, n, 28, 28).transpose(0, 2, 1, 3).reshape(n * 28, n * 28)

    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True, facecolor=SURFACE)
    ax.imshow(1 - sheet, cmap="gray", extent=extent, aspect="auto")
    style(ax, extent)
    ax.set_title("decoder φ(z) over the checked extent", color=INK, loc="left", fontsize=9.5)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


# -- main ----------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data_dir", type=Path, default=Path("experiments/mnist"))
    parser.add_argument("--cache_dir", type=Path, default=Path("experiments/mnist_islands"))
    parser.add_argument("--out", type=Path, default=Path("reports/mnist_islands"))
    parser.add_argument("--seed_classes", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--taus", type=float, nargs="+", default=[0.9, 0.99, 0.999])
    parser.add_argument("--clearance", type=float, default=0.0,
                        help="erode Omega* by this latent distance before labelling (Cl_eps)")
    parser.add_argument("--latent_dim", type=int, default=2, choices=[2],
                        help="the flood fill is exact only on a 2-d grid")
    parser.add_argument("--vae_epochs", type=int, default=30)
    parser.add_argument("--n_classifiers", type=int, default=3)
    parser.add_argument("--clf_epochs", type=int, default=3)
    parser.add_argument("--grid_res", type=int, default=400)
    parser.add_argument("--min_cells", type=int, default=20,
                        help="components smaller than this are reported but drawn as specks")
    parser.add_argument("--seed_min_share", type=float, default=0.01,
                        help="a component is part of S_0 if it holds this share of the seed-class codes")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    x, y = load_mnist(args.data_dir, train=True)
    x_test, y_test = load_mnist(args.data_dir, train=False)
    x, y = x.to(device), y.to(device)

    vae = get_vae(args.cache_dir, x, args.latent_dim, args.vae_epochs, args.seed, device)
    clfs = get_classifiers(args.cache_dir, x, y, args.n_classifiers, args.clf_epochs, args.seed, device)
    for k, clf in enumerate(clfs):
        acc = (batched(lambda b: clf(b).argmax(1), x_test.to(device)).cpu() == y_test).float().mean().item()
        print(f"  classifier {k}: test accuracy {acc:.4f}")

    codes = batched(lambda b: vae.encode(b)[0], x).cpu().numpy()
    labels = y.cpu().numpy()
    extent = latent_extent(codes)
    plot_manifold(vae, extent, device, args.out / "manifold.png")

    grid_cache = args.cache_dir / (f"grid_r{args.grid_res}_k{args.n_classifiers}_e{args.vae_epochs}x{args.clf_epochs}"
                                   f"_s{args.seed}.npz")
    cls, conf = classify_grid(vae, clfs, extent, args.grid_res, grid_cache, device)
    print(f"grid {args.grid_res}^2 over z1 [{extent[0]:.2f}, {extent[1]:.2f}], z2 [{extent[2]:.2f}, {extent[3]:.2f}]; "
          f"ensemble agrees on {(cls >= 0).mean():.1%} of it")

    fields = ["tau", "clearance", "component", "cells", "area", "majority", "class_mix", "touching", "data_share",
              "data_labels", "seed_code_share", "speck", "seed", "gap"]
    all_rows = []
    for tau in args.taus:
        lab, rows = label_islands(cls, conf, tau, extent, codes, labels, args.seed_classes, args.seed_min_share,
                                  args.min_cells, args.clearance)
        plot_islands(args, lab, rows, cls, conf, tau, extent, args.out / f"islands_tau{tau:g}.png")
        all_rows += [{"tau": tau, "clearance": args.clearance, **r} for r in rows]

        seed_rows = [r for r in rows if r["seed"]]
        islands = sorted((r for r in rows if not r["seed"] and not r["speck"]), key=lambda r: r["gap"])
        absorbed = sorted({k for r in seed_rows for k, _ in
                           (p.split(":") for p in r["class_mix"].split()) if int(k) not in args.seed_classes})
        print(f"\ntau={tau:g}: valid {(lab > 0).mean():.1%} of the grid, {len(rows)} components, "
              f"{len(seed_rows)} in S_0, {len(islands)} islands, {sum(r['speck'] for r in rows)} specks")
        if absorbed:
            print(f"  connected to S_0 by a valid path (not islands): classes {', '.join(absorbed)}")
        for r in islands:
            print(f"  island {r['majority']}  gap {r['gap']:.3f}  area {r['area']:.3f}  "
                  f"data {r['data_share']:.1%}  mix [{r['class_mix']}]")
        missing = sorted(set(range(10)) - set(args.seed_classes) - {r["majority"] for r in islands} - set(map(int, absorbed)))
        if missing:
            print(f"  no island at all: classes {missing}")

    path = args.out / "components.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()})
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main(build_parser().parse_args())
