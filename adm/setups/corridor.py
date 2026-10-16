from typing import Any, Optional

import torch
from diffusiongym import DDTensor, BaseModel, OptimalTransportScheduler, Scheduler, Environment
from diffusiongym.utils import train_base_model
from diffusiongym.base_models.one_dim_gmm import MLP
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pathlib import Path
import os

from .problem_setup import ProblemSetup
from adm.uncertainty import UncertaintyEstimator
from adm.utils import Batch


class CorridorProblemSetup(ProblemSetup[DDTensor]):
    """Two islands joined by a narrow channel, i.e., a $(w, \\ell)$ corridor.

    The valid region is $\\Omega_\\star = S_0 \\cup C \\cup A$, laid out along the x-axis:

        S_0 = [x0 - s, x0]           x [-s/2, s/2]   (source island, pre-training data)
        C   = [x0, x0 + l]           x [-w/2, w/2]   (corridor)
        A   = [x0 + l, x0 + l + s]   x [-s/2, s/2]   (target island)

    Every point of C lies within w/2 of the boundary of Omega and the geodesic distance through C
    from S_0 to A is l, so the width and length of the corridor are controlled directly. The
    position of S_0 does not depend on (w, l), so the same pre-trained base model can be shared
    across a sweep via `--corridor_base_ckpt`.

    With w = 0 there is no corridor, and S_0 and A are disconnected islands separated by a gap of l.
    `frac_between` in the metrics is the fraction of samples between the islands (in the corridor
    or the gap), which is a measure of spurious mass bridging the islands when w = 0.
    """

    def __init__(self, args: dict[str, Any], device: Optional[torch.device]=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.width = args.get("corridor_width", 0.2)
        self.length = args.get("corridor_length", 1.5)
        self.island_size = args.get("corridor_island_size", 1.0)
        self.x0 = args.get("corridor_x0", -1.5)
        self.solve_count = args.get("corridor_solve_count", 1)

        if self.width < 0 or self.length <= 0:
            raise ValueError(f"corridor width must be non-negative and length positive, got w={self.width}, l={self.length}")

        if self.width > self.island_size:
            raise ValueError(f"corridor width ({self.width}) cannot exceed island size ({self.island_size})")

        ckpt = args.get("corridor_base_ckpt")
        self._base_model = CorridorBaseModel(
            source_box=self.source_box,
            ckpt=Path(ckpt) if ckpt is not None else None,
            device=device,
        )

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        parser.add_argument("--corridor_width", type=float, default=0.2)
        parser.add_argument("--corridor_length", type=float, default=1.5)
        parser.add_argument("--corridor_island_size", type=float, default=1.0)
        parser.add_argument("--corridor_x0", type=float, default=-1.5)
        parser.add_argument("--corridor_solve_count", type=int, default=1, help="Valid samples in A needed to count as solved")
        parser.add_argument("--corridor_base_ckpt", type=str, default=None, help="Cache for the pre-trained base model")

    @property
    def base_model(self) -> BaseModel[DDTensor]:
        return self._base_model

    @property
    def source_box(self) -> tuple[float, float, float, float]:
        s = self.island_size
        return (self.x0 - s, self.x0, -s / 2, s / 2)

    @property
    def corridor_box(self) -> tuple[float, float, float, float]:
        return (self.x0, self.x0 + self.length, -self.width / 2, self.width / 2)

    @property
    def target_box(self) -> tuple[float, float, float, float]:
        s = self.island_size
        x1 = self.x0 + self.length
        return (x1, x1 + s, -s / 2, s / 2)

    @staticmethod
    def _in_box(xy: torch.Tensor, box: tuple[float, float, float, float]) -> torch.Tensor:
        xmin, xmax, ymin, ymax = box
        x, y = xy[:, 0], xy[:, 1]
        return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)

    def in_target(self, samples: DDTensor) -> torch.Tensor:
        return self._in_box(samples.data, self.target_box)

    def progress(self, samples: DDTensor) -> torch.Tensor:
        """Fraction of the corridor traversed, clipped to [0, 1]: 0 at the mouth, 1 at A."""
        return ((samples.data[:, 0] - self.x0) / self.length).clamp(0, 1)

    def validity(self, samples: DDTensor, kwargs: dict[str, Any]) -> torch.Tensor:
        xy = samples.data
        inside = self._in_box(xy, self.source_box) | self._in_box(xy, self.target_box)
        if self.width > 0:
            inside = inside | self._in_box(xy, self.corridor_box)
        return inside.bool()

    def is_solved(self, batch: Batch[DDTensor]) -> bool:
        hits = batch.valids.to(batch.samples.data.device) & self.in_target(batch.samples)
        return hits.sum().item() >= self.solve_count

    @property
    def feature_layer(self) -> str:
        return "input"

    def postprocess_features(self, latents: DDTensor, feats: DDTensor) -> torch.Tensor:
        return feats.data

    def compute_metrics(self, samples: DDTensor, kwargs: dict) -> dict[str, float]:
        valid = self.validity(samples, kwargs).to(samples.data.device)
        prog = self.progress(samples)[valid]
        return {
            "frac_in_target": (valid & self.in_target(samples)).float().mean().item(),
            "max_progress": prog.max().item() if len(prog) > 0 else 0.0,
            "q99_progress": prog.quantile(0.99).item() if len(prog) > 0 else 0.0,
            "frac_between": self._in_box(samples.data, (self.x0, self.x0 + self.length, -self.island_size / 2, self.island_size / 2)).float().mean().item(),
        }

    @property
    def view_extent(self) -> tuple[float, float, float, float]:
        s = self.island_size
        xmin = self.x0 - s - 0.5
        xmax = self.x0 + self.length + s + 0.5
        return (xmin, xmax, -s / 2 - 0.5, s / 2 + 0.5)

    def draw_domain(self, ax):
        for box in (self.source_box, self.corridor_box, self.target_box):
            xmin, xmax, ymin, ymax = box
            ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor="black", lw=0.8, alpha=0.6))

    @torch.no_grad()
    def visualize_sample(
        self,
        env: Environment[DDTensor],
        uncertainty: UncertaintyEstimator[DDTensor],
        batch: Batch[DDTensor],
    ) -> Figure:
        xmin, xmax, ymin, ymax = self.view_extent

        x = torch.linspace(xmin, xmax, 200)
        y = torch.linspace(ymin, ymax, 60)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        XY = DDTensor(torch.stack([X.flatten(), Y.flatten()], dim=-1)).to(self.device)
        _, Z = uncertainty.mean_and_uncertainty(XY)
        Z = Z.reshape(X.shape)

        fig, axes = plt.subplots(2, 1, figsize=(6, 4.5), constrained_layout=True)
        im = axes[0].imshow(Z.T.cpu(), extent=(xmin, xmax, ymin, ymax), origin="lower", cmap="YlGn", aspect="equal")
        fig.colorbar(im, ax=axes.tolist(), shrink=0.6, label="uncertainty")
        axes[0].set_title(f"Uncertainty and batch (w={self.width:g}, l={self.length:g})")

        data = batch.samples.data.cpu()
        valid = batch.valids.cpu()
        axes[0].scatter(data[valid][:, 0], data[valid][:, 1], s=3, label="valid")
        axes[0].scatter(data[~valid][:, 0], data[~valid][:, 1], s=3, label="invalid")

        many = env.sample(4096, pbar=False).sample.data.cpu()
        axes[1].scatter(many[:, 0], many[:, 1], s=1, alpha=0.3, color="black")
        axes[1].set_title("Model samples")

        for ax in axes:
            self.draw_domain(ax)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect("equal")

        return fig

    def save_samples(self, samples: DDTensor, kwargs: dict, dir: Path) -> bool:
        torch.save(samples.data.cpu(), dir / "samples.pt")
        return True

    def load_samples(self, dir: Path) -> tuple[DDTensor, dict]:
        return DDTensor(torch.load(dir / "samples.pt")), {}


class CorridorBaseModel(BaseModel[DDTensor]):
    """Flow model pre-trained on uniform samples from the source island $S_0$."""

    output_type = "velocity"

    def __init__(
        self,
        source_box: tuple[float, float, float, float],
        ckpt: Optional[Path] = None,
        num_data: int = 2048,
        steps: int = 2500,
        device: Optional[torch.device]=None,
    ):
        super().__init__(device)
        device = self.device

        self._scheduler = OptimalTransportScheduler()
        self.model = MLP(2, 2).to(device)

        if ckpt is not None and ckpt.exists():
            self.load_state_dict(torch.load(ckpt, map_location=device))
            return

        xmin, xmax, ymin, ymax = source_box
        lo = torch.tensor([xmin, ymin])
        hi = torch.tensor([xmax, ymax])
        data = lo + (hi - lo) * torch.rand(num_data, 2)
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        train_base_model(self, opt, [DDTensor(data)], steps=steps, batch_size=256, pbar=False)

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
