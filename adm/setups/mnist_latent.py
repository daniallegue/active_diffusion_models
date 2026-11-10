from typing import Any, Optional
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusiongym import DDTensor, BaseModel, OptimalTransportScheduler, Scheduler, Environment
from diffusiongym.utils import train_base_model
from diffusiongym.base_models.one_dim_gmm import MLP
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy import ndimage

from .problem_setup import ProblemSetup
from adm.uncertainty import UncertaintyEstimator
from adm.utils import Batch

EIGHT_CONNECTED = np.ones((3, 3), dtype=int)


# -- the geometry phi and the verifier's classifiers ---------------------------------------------

class VAE(nn.Module):
    def __init__(self, latent_dim: int = 2, hidden: int = 512):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, hidden), nn.ReLU(), nn.Linear(hidden, hidden // 2), nn.ReLU())
        self.mu = nn.Linear(hidden // 2, latent_dim)
        self.logvar = nn.Linear(hidden // 2, latent_dim)
        self.dec = nn.Sequential(nn.Linear(latent_dim, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, hidden),
                                 nn.ReLU(), nn.Linear(hidden, 784))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x.flatten(1))
        return self.mu(h), self.logvar(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.dec(z)).view(-1, 1, 28, 28)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        logits = self.dec(z)
        rec = F.binary_cross_entropy_with_logits(logits, x.flatten(1), reduction="sum")
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
        return (rec + kl) / len(x)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_mnist(data_dir: Path, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    from torchvision.datasets import MNIST

    ds = MNIST(str(data_dir), train=train, download=True)
    return ds.data.float().unsqueeze(1) / 255.0, ds.targets.clone()


@torch.no_grad()
def batched(fn, x: torch.Tensor, batch_size: int = 4096) -> torch.Tensor:
    return torch.cat([fn(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])


def fit(model: nn.Module, step, n: int, epochs: int, batch_size: int, lr: float, device: torch.device,
        name: str):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, batch_size):
            loss = step(perm[i:i + batch_size])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(perm[i:i + batch_size])
        print(f"  {name} epoch {epoch + 1}/{epochs}: loss {total / n:.4f}")
    model.eval()


def get_vae(cache_dir: Path, x: torch.Tensor, latent_dim: int, epochs: int, seed: int, device: torch.device) -> VAE:
    path = cache_dir / f"vae_d{latent_dim}_e{epochs}_s{seed}.pt"
    vae = VAE(latent_dim).to(device)
    if path.exists():
        vae.load_state_dict(torch.load(path, map_location=device))
        vae.eval()
        return vae

    print(f"training VAE -> {path}")
    torch.manual_seed(seed)
    fit(vae, lambda idx: vae.loss(x[idx]), len(x), epochs, 128, 1e-3, device, "vae")
    torch.save(vae.state_dict(), path)
    return vae


def get_classifiers(cache_dir: Path, x: torch.Tensor, y: torch.Tensor, n: int, epochs: int, seed: int,
                    device: torch.device) -> list[Classifier]:
    clfs = []
    for k in range(n):
        path = cache_dir / f"clf{k}_e{epochs}_s{seed}.pt"
        clf = Classifier().to(device)
        if path.exists():
            clf.load_state_dict(torch.load(path, map_location=device))
            clf.eval()
        else:
            print(f"training classifier {k} -> {path}")
            torch.manual_seed(seed * 1000 + k)
            fit(clf, lambda idx: F.cross_entropy(clf(x[idx]), y[idx]), len(x), epochs, 128, 1e-3, device, f"clf{k}")
            torch.save(clf.state_dict(), path)
        clfs.append(clf)
    return clfs


# -- the verifier on a grid ----------------------------------------------------------------------

Extent = tuple[float, float, float, float]   # xmin, xmax, ymin, ymax


def latent_extent(codes: np.ndarray, pad: float = 0.1) -> Extent:
    """The box holding all but the outermost 0.1% of the training codes, padded. Beyond it the
    decoder extrapolates, and a confident classifier there says nothing about digits."""
    lo, hi = np.quantile(codes, 0.001, axis=0), np.quantile(codes, 0.999, axis=0)
    span = hi - lo
    lo, hi = lo - pad * span, hi + pad * span
    return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])


@torch.no_grad()
def classify_grid(vae: VAE, clfs: list[Classifier], extent: Extent, res: int, cache: Path,
                  device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Per grid cell, the class the ensemble agrees on (-1 where it does not) and the lowest member
    confidence in that class. Indexed [iy, ix], so imshow with origin="lower" puts y up."""
    if cache.exists():
        d = np.load(cache)
        if np.allclose(d["extent"], extent):
            return d["cls"], d["conf"]

    xmin, xmax, ymin, ymax = extent
    xs = torch.linspace(xmin, xmax, res)
    ys = torch.linspace(ymin, ymax, res)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    z = torch.stack([X.flatten(), Y.flatten()], dim=1).to(device)

    imgs = batched(vae.decode, z)
    probs = torch.stack([batched(lambda b: F.softmax(clf(b), dim=1), imgs) for clf in clfs])   # K, N, 10
    votes = probs.argmax(2)                                                                      # K, N
    cls = votes[0]
    agreed = (votes == cls).all(0)
    conf = probs.gather(2, cls.view(1, -1, 1).expand(len(clfs), -1, 1)).squeeze(2).min(0).values
    cls = torch.where(agreed, cls, torch.full_like(cls, -1))

    cls, conf = cls.view(res, res).cpu().numpy(), conf.view(res, res).cpu().numpy()
    np.savez(cache, cls=cls, conf=conf, extent=np.array(extent))
    return cls, conf


def to_cells(z: np.ndarray, extent: Extent, res: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax = extent
    ix = np.round((z[:, 0] - xmin) / (xmax - xmin) * (res - 1)).astype(int)
    iy = np.round((z[:, 1] - ymin) / (ymax - ymin) * (res - 1)).astype(int)
    inside = (ix >= 0) & (ix < res) & (iy >= 0) & (iy < res)
    return iy.clip(0, res - 1), ix.clip(0, res - 1), inside


def label_islands(cls: np.ndarray, conf: np.ndarray, tau: float, extent: Extent, codes: np.ndarray,
                  labels: np.ndarray, seed_classes: list[int], seed_min_share: float = 0.01,
                  min_cells: int = 20, clearance: float = 0.0) -> tuple[np.ndarray, list[dict]]:
    """The components of Omega* at tau, as a label grid and one row per component.

    S_0 is the union of the components holding at least `seed_min_share` of the seed-class codes,
    which is where the base model's mass will be. The gap of an island is its Euclidean distance to
    S_0 in Z. 8-connectivity, so a diagonal pinch counts as connected: errs towards joining islands.
    """
    xmin, xmax, ymin, ymax = extent
    res = cls.shape[0]
    dx, dy = (xmax - xmin) / (res - 1), (ymax - ymin) / (res - 1)

    valid = (cls >= 0) & (conf >= tau)
    if clearance > 0:
        valid = ndimage.distance_transform_edt(valid, sampling=(dy, dx)) >= clearance

    lab, n = ndimage.label(valid, structure=EIGHT_CONNECTED)
    iy, ix, inside = to_cells(codes, extent, res)
    code_comp = np.where(inside, lab[iy, ix], 0)
    is_seed_code = np.isin(labels, seed_classes)

    rows = []
    for c in range(1, n + 1):
        mask = lab == c
        cells = int(mask.sum())
        decoded = np.bincount(cls[mask], minlength=10) / cells
        mine = code_comp == c
        true = np.bincount(labels[mine], minlength=10)
        rows.append({
            "component": c,
            "cells": cells,
            "area": cells * dx * dy,
            "majority": int(decoded.argmax()),
            "class_mix": " ".join(f"{k}:{decoded[k]:.2f}" for k in np.argsort(-decoded) if decoded[k] >= 0.01),
            "touching": int((decoded >= 0.01).sum() > 1),
            "data_share": float(mine.mean()),
            "data_labels": " ".join(f"{k}:{true[k]}" for k in np.argsort(-true) if true[k] > 0),
            "seed_code_share": float((mine & is_seed_code).sum() / max(is_seed_code.sum(), 1)),
            "speck": int(cells < min_cells),
        })

    for r in rows:
        r["seed"] = int(r["seed_code_share"] >= seed_min_share)

    s0 = np.isin(lab, [r["component"] for r in rows if r["seed"]])
    dist = ndimage.distance_transform_edt(~s0, sampling=(dy, dx)) if s0.any() else np.full(lab.shape, np.inf)
    for r in rows:
        r["gap"] = 0.0 if r["seed"] else float(dist[lab == r["component"]].min())

    return lab, rows


# -- the problem ---------------------------------------------------------------------------------

class MNISTLatentProblemSetup(ProblemSetup[DDTensor]):
    """Islands of confidently classified digits in a 2-d VAE latent; S_0 holds the seed classes."""

    def __init__(self, args: dict[str, Any], device: Optional[torch.device] = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.tau = args.get("mnist_tau", 0.99)
        self.seed_classes = list(args.get("mnist_seed_classes", [1, 7]))
        cache_dir = Path(args.get("mnist_cache_dir", "experiments/mnist_islands"))
        data_dir = Path(args.get("mnist_data_dir", "experiments/mnist"))
        res = args.get("mnist_grid_res", 400)
        vae_epochs, n_clf, clf_epochs = args.get("mnist_vae_epochs", 30), args.get("mnist_n_classifiers", 3), \
            args.get("mnist_clf_epochs", 3)
        asset_seed = args.get("mnist_asset_seed", 0)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # The geometry and the verifier are fixed assets, shared by every run: they are trained once
        # (by the check or the pretrain step) and loaded here, always on the CPU-sized MLPs
        x, y = load_mnist(data_dir, train=True)
        self.vae = get_vae(cache_dir, x.to(device), 2, vae_epochs, asset_seed, device)
        self.vae.requires_grad_(False)
        codes = batched(lambda b: self.vae.encode(b)[0], x.to(device)).cpu().numpy()
        labels = y.numpy()
        self.extent = latent_extent(codes)
        grid_cache = cache_dir / f"grid_r{res}_k{n_clf}_e{vae_epochs}x{clf_epochs}_s{asset_seed}.npz"
        if grid_cache.exists():
            clfs = []
        else:
            clfs = get_classifiers(cache_dir, x.to(device), y.to(device), n_clf, clf_epochs, asset_seed, device)
        self.cls, self.conf = classify_grid(self.vae, clfs, self.extent, res, grid_cache, device)
        self.lab, self.components = label_islands(
            self.cls, self.conf, self.tau, self.extent, codes, labels, self.seed_classes,
            args.get("mnist_seed_min_share", 0.01), args.get("mnist_min_cells", 20),
        )
        self.res = res

        # Lookup tables indexed by component id; 0 is the rejected set
        n = len(self.components) + 1
        self._is_seed = torch.zeros(n, dtype=torch.bool)
        self._is_island = torch.zeros(n, dtype=torch.bool)
        self._majority = torch.full((n,), -1, dtype=torch.long)
        for r in self.components:
            self._is_seed[r["component"]] = bool(r["seed"])
            self._is_island[r["component"]] = not r["seed"] and not r["speck"]
            self._majority[r["component"]] = r["majority"]
        self._lab = torch.from_numpy(self.lab).long()

        # The base model sees only the seed codes that lie in S_0, so its support is S_0 and nothing
        # it samples at the start is already an island
        comp = self.component_of(torch.from_numpy(codes))
        in_s0 = self._is_seed[comp].numpy() & np.isin(labels, self.seed_classes)
        self.seed_codes = torch.from_numpy(codes[in_s0])

        ckpt = args.get("mnist_base_ckpt")
        self._base_model = MNISTLatentBaseModel(
            self.seed_codes, ckpt=Path(ckpt) if ckpt is not None else None, device=device,
        )

        self._queried = 0
        self._first_hit: dict[int, int] = {}
        self._trace: list[dict[str, torch.Tensor]] = []

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        parser.add_argument("--mnist_tau", type=float, default=0.99, help="ensemble confidence the verifier needs")
        parser.add_argument("--mnist_seed_classes", type=int, nargs="+", default=[1, 7])
        parser.add_argument("--mnist_cache_dir", type=str, default="experiments/mnist_islands")
        parser.add_argument("--mnist_data_dir", type=str, default="experiments/mnist")
        parser.add_argument("--mnist_grid_res", type=int, default=400)
        parser.add_argument("--mnist_vae_epochs", type=int, default=30)
        parser.add_argument("--mnist_n_classifiers", type=int, default=3)
        parser.add_argument("--mnist_clf_epochs", type=int, default=3)
        parser.add_argument("--mnist_asset_seed", type=int, default=0)
        parser.add_argument("--mnist_seed_min_share", type=float, default=0.01)
        parser.add_argument("--mnist_min_cells", type=int, default=20)
        parser.add_argument("--mnist_base_ckpt", type=str, default=None, help="Cache for the pre-trained base model")

    @property
    def base_model(self) -> BaseModel[DDTensor]:
        return self._base_model

    @property
    def islands(self) -> list[dict]:
        return [r for r in self.components if not r["seed"] and not r["speck"]]

    def component_of(self, z: torch.Tensor) -> torch.Tensor:
        """The component of Omega* each latent lies in, 0 where the verifier rejects it."""
        iy, ix, inside = to_cells(z.detach().cpu().numpy(), self.extent, self.res)
        comp = self._lab[torch.from_numpy(iy), torch.from_numpy(ix)]
        return torch.where(torch.from_numpy(inside), comp, torch.zeros_like(comp))

    def validity(self, samples: DDTensor, kwargs: dict[str, Any]) -> torch.Tensor:
        return (self.component_of(samples.data) > 0).to(samples.data.device)

    def in_target(self, samples: DDTensor) -> torch.Tensor:
        return self._is_island[self.component_of(samples.data)].to(samples.data.device)

    def is_solved(self, batch: Batch[DDTensor]) -> bool:
        """Records the round's queries, and counts as solved once any island outside S_0 is hit.

        Both loops call this exactly once per round on the queried batch, which makes it the one
        place the query trace can be kept without touching either loop.
        """
        comp = self.component_of(batch.samples.data)
        valid = batch.valids.cpu().bool()
        self._queried += len(comp)
        for c in comp[valid & self._is_island[comp]].unique().tolist():
            self._first_hit.setdefault(c, self._queried)

        self._trace.append({"z": batch.samples.data.detach().cpu(), "valid": valid, "comp": comp,
                            "queried": torch.tensor(self._queried)})
        return len(self._first_hit) > 0

    def hit_summary(self) -> dict[str, Any]:
        """Per island: its class, its gap to S_0 and the queries to its first hit (None if never)."""
        islands = [{"component": r["component"], "class": r["majority"], "gap": r["gap"], "area": r["area"],
                    "data_share": r["data_share"], "first_hit": self._first_hit.get(r["component"])}
                   for r in self.islands]
        new = sorted(set(range(10)) - set(self.seed_classes))
        hit_classes = sorted({i["class"] for i in islands if i["first_hit"] is not None and i["class"] in new})
        return {"islands": islands, "islands_hit": sum(i["first_hit"] is not None for i in islands),
                "num_islands": len(islands), "classes_hit": hit_classes, "num_new_classes": len(new)}

    def save_trace(self, path: Path):
        torch.save(self._trace, path)

    @property
    def feature_layer(self) -> str:
        return "input"

    def postprocess_features(self, latents: DDTensor, feats: DDTensor) -> torch.Tensor:
        return feats.data

    def compute_metrics(self, samples: DDTensor, kwargs: dict) -> dict[str, float]:
        comp = self.component_of(samples.data)
        on_island = self._is_island[comp]
        metrics = {
            "frac_valid": (comp > 0).float().mean().item(),
            "frac_in_s0": self._is_seed[comp].float().mean().item(),
            "frac_in_target": on_island.float().mean().item(),
            "islands_with_mass": len(comp[on_island].unique()),
        }
        cls = self._majority[comp]
        for k in sorted(set(range(10)) - set(self.seed_classes)):
            metrics[f"frac_class{k}"] = (on_island & (cls == k)).float().mean().item()
        return metrics

    def draw_islands(self, ax):
        """S_0 in blue, the islands in orange, the rejected set pale, as in the check's figures."""
        from matplotlib.colors import to_rgb

        rgb = np.empty(self.lab.shape + (3,))
        rgb[:] = to_rgb("#ecebe7")
        rgb[np.isin(self.lab, [r["component"] for r in self.components if r["seed"]])] = to_rgb("#9ec5f4")
        rgb[np.isin(self.lab, [r["component"] for r in self.islands])] = to_rgb("#f5b69a")
        ax.imshow(rgb, extent=self.extent, origin="lower", aspect="auto", interpolation="nearest")

    @torch.no_grad()
    def visualize_sample(
        self,
        env: Environment[DDTensor],
        uncertainty: UncertaintyEstimator[DDTensor],
        batch: Batch[DDTensor],
    ) -> Figure:
        fig, ax = plt.subplots(figsize=(5, 4.5), constrained_layout=True)
        self.draw_islands(ax)
        many = env.sample(2048, pbar=False).sample.data.cpu()
        ax.scatter(many[:, 0], many[:, 1], s=1, alpha=0.3, color="black")
        data, valid = batch.samples.data.cpu(), batch.valids.cpu().bool()
        ax.scatter(data[valid][:, 0], data[valid][:, 1], s=6, color="#1baf7a")
        ax.scatter(data[~valid][:, 0], data[~valid][:, 1], s=6, color="#e34948", marker="x")
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])
        return fig

    def save_samples(self, samples: DDTensor, kwargs: dict, dir: Path) -> bool:
        torch.save(samples.data.cpu(), dir / "samples.pt")
        return True

    def load_samples(self, dir: Path) -> tuple[DDTensor, dict]:
        return DDTensor(torch.load(dir / "samples.pt")), {}


class MNISTLatentBaseModel(BaseModel[DDTensor]):
    """Flow model pre-trained on the seed-class latent codes that lie in $S_0$."""

    output_type = "velocity"

    def __init__(
        self,
        codes: torch.Tensor,
        ckpt: Optional[Path] = None,
        steps: int = 5000,
        device: Optional[torch.device] = None,
    ):
        super().__init__(device)
        device = self.device

        self._scheduler = OptimalTransportScheduler()
        self.model = MLP(2, 2).to(device)

        if ckpt is not None and ckpt.exists():
            self.load_state_dict(torch.load(ckpt, map_location=device))
            return

        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        train_base_model(self, opt, [DDTensor(codes.float())], steps=steps, batch_size=256, pbar=False)

        if ckpt is not None:
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            tmp = ckpt.with_name(f"{ckpt.name}.{os.getpid()}.tmp")
            torch.save(self.state_dict(), tmp)
            os.replace(tmp, ckpt)

    @property
    def scheduler(self) -> Scheduler[DDTensor]:
        return self._scheduler

    def sample_p0(self, n: int, **kwargs: Any) -> tuple[DDTensor, dict[str, Any]]:
        return DDTensor(torch.randn(n, 2, device=self.device)), {}

    def forward(self, x: DDTensor, t: torch.Tensor, **kwargs: Any) -> DDTensor:
        return DDTensor(self.model(x.data, t))
