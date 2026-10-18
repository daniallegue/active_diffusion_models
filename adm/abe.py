"""Active Bridge Expansion (ABE): the Schrödinger bridge variant of ActFlow.
"""

from typing import Any, Callable, Literal, Optional
from dataclasses import dataclass, replace
import argparse
import copy
import math

import torch
import torch.nn.functional as F
from diffusiongym import D, DDTensor, DummyReward, BaseModel, Environment, Sample, construct_env
from diffusiongym.utils import train_base_model, dict_to_device
from diffusiongym.base_models.one_dim_gmm import MLP

from .explore import ExploreLoop, build_loop, build_parser
from .utils import Batch, write_video


@dataclass(frozen=True)
class ABEConfig:
    # Reference process Q and dispersal
    step_size: float = 0.05          # eps, diffusion coefficient of Q and entropic OT regularization
    disperse_time: float = 1.0       # tau_t, time that the pool is simulated under Q
    reference: Literal["gaussian", "student", "levy", "poisson"] = "gaussian"
    reference_df: float = 2.0        # degrees of freedom of "student"
    reference_alpha: float = 1.5     # stability index of "levy", in (0, 2]
    jump_rate: float = 0.2           # Poisson rate of jumps per unit time for "poisson"
    jump_scale: float = 1.0          # standard deviation of each jump for "poisson"
    max_jump: float = 5.0            # cap on the norm of a dispersal step, for heavy tails

    # Dispersal from the frontier of the verified positives
    frontier_frac: float = 0.5       # fraction of the pool that starts at positives instead of mu_theta

    # Surrogate and tilt
    acquisition: Literal["uncertainty", "ucb"] = "uncertainty"
    feasibility_weight: float = 0.25 # lambda, x is feasible iff mu(x) + lambda sigma(x) >= 0
    standardize_labels: bool = False # otherwise, validity labels are centred to +-0.5
    tilt: float = 5.0                # beta_t
    kappa: float = 1.0               # exploration weight of "ucb"
    ess_min: float = 0.1             # floor on ESS(w) / N

    # Target and bridge
    pool_size: int = 1024            # N
    smooth_bandwidth: float = 0.02   # bandwidth h of Smooth
    bridge: Literal["cfm", "sb", "none"] = "cfm"
    bridge_steps: int = 500
    bridge_batch_size: int = 256
    bridge_lr: float = 1e-3
    sinkhorn_iters: int = 100

    # Proposal, selection, and querying
    proposal_size: int = 256         # B'
    query_budget: int = 16           # B
    exclusion_radius: float = 0.05   # delta

    # Distillation
    distill_filter: Literal["lcb", "none"] = "lcb"
    safety_weight: float = 0.1       # lambda_s, x is confidently valid iff mu(x) - lambda_s sigma(x) >= 0
    mix_weight: float = 0.5          # alpha
    trust_radius: float = 0.1        # rho
    trust_direction: Literal["reverse", "forward"] = "reverse"  # KL(mu_theta || mu_theta') or KL(mu_theta' || mu_theta)
    distill_size: int = 1024
    distill_check_every: int = 50
    trust_samples: int = 256         # paths for the Girsanov estimate of the trust region

    # Diagnostics
    save_rounds: bool = False        # save pool, proposals, selection, and model samples per round

    def __post_init__(self):
        if not (0 < self.step_size < 1):
            raise ValueError(f"step_size must be in (0, 1), got {self.step_size}")

        if not (0 <= self.mix_weight <= 1):
            raise ValueError(f"mix_weight must be in [0, 1], got {self.mix_weight}")

        if not (0 <= self.frontier_frac <= 1):
            raise ValueError(f"frontier_frac must be in [0, 1], got {self.frontier_frac}")

        if not (0 < self.ess_min <= 1):
            raise ValueError(f"ess_min must be in (0, 1], got {self.ess_min}")

        if self.query_budget > self.proposal_size:
            raise ValueError(f"query_budget ({self.query_budget}) cannot exceed proposal_size ({self.proposal_size})")

    @staticmethod
    def construct_from_args(args: argparse.Namespace | dict) -> "ABEConfig":
        if isinstance(args, argparse.Namespace):
            args = vars(args)

        # CLI arguments are the field names prefixed with `abe_`
        config_fields = set(ABEConfig.__dataclass_fields__)
        clean_kwargs = {k[4:]: v for k, v in args.items() if k.startswith("abe_") and k[4:] in config_fields}
        return ABEConfig(**clean_kwargs)


class ABELoop(ExploreLoop[D]):
    """ABE on any latent type D. Particles are always kept as latents of type D, and the problem setup
    provides the operations that depend on the representation: `disperse` (line 4), and
    `distance_features` and `distance` (diverse selection and the trust region). The cfm and sb
    bridges need fixed-size tensors; `bridge="none"` proposes from mu* directly and works for any D,
    and is used automatically for other latents, e.g., molecular graphs."""

    def __init__(self, *args: Any, abe_config: ABEConfig, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.abe = abe_config

        x, sample_kwargs = self.base_model.sample_p0(1)
        if len(sample_kwargs) > 0:
            raise ValueError("ABE only supports unconditional problem setups")

        if self.abe.bridge != "none" and not isinstance(x, DDTensor):
            self.logger.warning(f"the {self.abe.bridge} bridge requires DDTensor latents, got {type(x).__name__}; using bridge='none'")
            self.abe = replace(self.abe, bridge="none")

        self._surrogate_fitted = False
        if self.abe.bridge != "none":
            self.latent_shape = x.data.shape[1:]
            dim = x.data[0].numel()
            self.bridge = MLP(dim, dim).to(self.base_model.device)
            self.bridge_opt = torch.optim.Adam(self.bridge.parameters(), lr=self.abe.bridge_lr)

        # ABE has different phases than ActFlow, so we rewrite the timings header
        self.timing_heads = [
            "iteration", "eval", "disperse", "bridge", "propose", "select", "query", "visualize",
            "finetune", "uncertainty_update", "checkpoint",
        ]
        with open(self.timing_path, "w", newline="") as f:
            f.write(",".join(self.timing_heads) + "\n")

    @property
    def device(self) -> torch.device:
        return self.base_model.device

    def _flat(self, x: DDTensor) -> torch.Tensor:
        return x.data.flatten(1)

    def _unflat(self, x: torch.Tensor) -> DDTensor:
        return DDTensor(x.reshape(-1, *self.latent_shape))

    @torch.no_grad()
    def sample_model(self, n: int) -> D:
        """Draws n samples from the current flow model mu_theta as latents."""
        if n == 0:
            return take(self.sample_model(1), torch.arange(0))

        sample = self.env.batch_sample(n, self.config.sample_batch_size)
        return self.problem.postprocess_latents(Batch.from_sample(sample)).to(self.device)

    def disperse(self, x: D, scale: float, reference: Optional[str] = None) -> D:
        """Disperses latents under the reference Q with the given scale (line 4). The problem setup
        decides how a displacement acts on its latents."""
        if len(x) == 0 or scale <= 0:
            return x

        def noise(n: int, d: int) -> torch.Tensor:
            return sample_reference(n, d, scale, self.abe, self.device, reference)

        return self.problem.disperse(self.env, x, scale, noise)

    @torch.no_grad()
    def features(self, x: D) -> torch.Tensor:
        """Distance features of latents: the problem setup's, or else the uncertainty estimator's."""
        if len(x) == 0:
            return torch.zeros(0, 0, device=self.device)

        feats = self.problem.distance_features(x)
        if feats is None:
            feats = self.uncertainty._get_feats(x)

        return feats.to(self.device)

    def distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.problem.distance(a, b)

    @torch.no_grad()
    def acquisition(self, x: D, has_data: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Acquisition score and feasibility mask of the surrogate. For "uncertainty", the score is
        sigma(x), and x is feasible unless it is confidently invalid, i.e., mu(x) + lambda sigma(x) < 0.
        For "ucb", the score is mu(x) + kappa sigma(x) and everything is feasible. Before any data or
        without uncertainty, the score is zero everywhere."""
        feasible = torch.ones(len(x), dtype=torch.bool, device=self.device)
        if not has_data or self.config.no_uncertainty:
            return torch.zeros(len(x), device=self.device), feasible

        mean, uncert = self.uncertainty.mean_and_uncertainty(x)
        mean, uncert = mean.to(self.device), uncert.to(self.device)
        if self.abe.acquisition == "ucb":
            return mean + self.abe.kappa * uncert, feasible

        return uncert, mean + self.abe.feasibility_weight * uncert >= 0

    @torch.no_grad()
    def confidently_valid(self, x: D) -> torch.Tensor:
        """Lower-confidence-bound rule mu(x) - lambda_s sigma(x) >= 0 of the surrogate. Without a
        fitted surrogate, nothing is confidently valid."""
        if len(x) == 0 or not self._surrogate_fitted:
            return torch.zeros(len(x), dtype=torch.bool, device=self.device)

        mean, uncert = self.uncertainty.mean_and_uncertainty(x)
        return (mean - self.abe.safety_weight * uncert >= 0).to(self.device)

    def update_uncertainty_estimator(self, batches: list[Batch[D]]):
        """Fits the surrogate on validity labels. Unless `standardize_labels`, these are centred to
        +-0.5 without rescaling, such that the zero prior mean of a GP means "unknown" and a single
        invalid point does not outweigh many valid ones."""
        self._surrogate_fitted = True
        if self.abe.standardize_labels:
            return super().update_uncertainty_estimator(batches)

        self.uncertainty.set_data(
            [b.latents for b in batches],
            [b.valids.float() - 0.5 for b in batches],
            [b.kwargs for b in batches],
            standardize=False,
        )

    def positives(self, batches: list[Batch[D]]) -> D:
        """All verified positives D^+ so far."""
        return cat([take(b.latents, b.valids.nonzero().squeeze(1)) for b in batches]).to(self.device)

    def fit_bridge(self, source: D, target: D, target_weights: torch.Tensor) -> float:
        """Fits the bridge between `source` and the smoothed weighted `target` (line 8)."""
        if self.abe.bridge == "none":
            return 0.0

        source, target = self._flat(source), self._flat(target)  # type: ignore
        eps = self.abe.step_size
        bs = self.abe.bridge_batch_size
        loss_sum = 0.0

        self.bridge.train()
        for _ in range(self.abe.bridge_steps):
            x0 = source[torch.randint(len(source), (bs,), device=self.device)]
            x1 = target[torch.multinomial(target_weights, bs, replacement=True)]
            x1 = x1 + self.abe.smooth_bandwidth * torch.randn_like(x1)

            if self.abe.bridge == "cfm":
                # Flow matching on straight paths between independently coupled pairs
                t = torch.rand(bs, 1, device=self.device)
                xt = (1 - t) * x0 + t * x1
                drift = x1 - x0
            else:
                # Static Schrödinger bridge on the minibatch
                with torch.no_grad():
                    cost = 0.5 * torch.cdist(x0, x1) ** 2
                    log_plan = log_sinkhorn(cost, eps, self.abe.sinkhorn_iters)
                    pairs = torch.multinomial(log_plan.flatten().softmax(0), bs, replacement=True)
                    x0, x1 = x0[pairs // bs], x1[pairs % bs]

                # Bridge matching on Brownian bridges of Q between the coupled pairs
                t = 0.99 * torch.rand(bs, 1, device=self.device)
                xt = (1 - t) * x0 + t * x1 + torch.sqrt(eps * t * (1 - t)) * torch.randn_like(x0)
                drift = (x1 - xt) / (1 - t)

            loss = F.mse_loss(self.bridge(xt, t.squeeze(1)), drift)
            self.bridge_opt.zero_grad()
            loss.backward()
            self.bridge_opt.step()
            loss_sum += loss.item()

        self.bridge.eval()
        return loss_sum / max(1, self.abe.bridge_steps)

    @torch.no_grad()
    def simulate_bridge(self, x0: DDTensor) -> DDTensor:
        """Simulates the fitted bridge from x0 at t = 0 to its endpoint at t = 1, as an ODE for `cfm`
        and as an SDE with the noise of Q for `sb`."""
        eps = self.abe.step_size if self.abe.bridge == "sb" else 0.0
        num_steps = self.config.num_steps
        dt = 1.0 / num_steps

        x = self._flat(x0).clone()
        for k in range(num_steps):
            t = torch.full((len(x),), k * dt, device=self.device)
            x = x + dt * self.bridge(x, t) + math.sqrt(eps * dt) * torch.randn_like(x)

        return self._unflat(x)

    def propose(self, n: int, target: D, target_weights: torch.Tensor) -> D:
        """Draws n samples of the bridge endpoint M_1 (line 9). Without a bridge, M_1 = mu*: the
        target is resampled and smoothed with a Gaussian dispersal of bandwidth h."""
        if self.abe.bridge != "none":
            return self.simulate_bridge(self.sample_model(n))  # type: ignore

        x = take(target, torch.multinomial(target_weights, n, replacement=True))
        return self.disperse(x, self.abe.smooth_bandwidth, reference="gaussian")

    def distill(self, bridge_endpoints: D, positives: D) -> dict[str, Any]:
        """Distills alpha M_1 + (1 - alpha) mu_theta into the flow model for `ft_steps`,
        with a trust region of radius rho: training stops early once a chunk moves the model further
        than rho, and that chunk is rolled back. The distance moved is the Girsanov KL between the
        path measures of the old and new model (`path_kl`), by default KL(P_old || P_new), which only
        penalizes losing mass that the old model had.

        With the "lcb" filter, only verified positives and confidently valid points are distilled:
        the bridge part is resampled from the confidently valid endpoints and the positives, and the
        model part from the confidently valid model samples (or the bridge part if there are none)."""
        n = self.abe.distill_size
        anchor = self.sample_model(n - len(bridge_endpoints))
        info: dict[str, Any] = {}

        if self.abe.distill_filter == "lcb":
            safe_endpoints = self.confidently_valid(bridge_endpoints)
            safe_anchor = self.confidently_valid(anchor)
            info["safe_ends"] = safe_endpoints.float().mean().item() if len(bridge_endpoints) > 0 else 0.0
            info["safe_anchor"] = safe_anchor.float().mean().item() if len(anchor) > 0 else 0.0

            bridge_pool = cat([take(bridge_endpoints, safe_endpoints.nonzero().squeeze(1)), positives])
            anchor_pool = take(anchor, safe_anchor.nonzero().squeeze(1)) if safe_anchor.any() else bridge_pool
            if len(bridge_pool) == 0:
                bridge_pool = anchor_pool

            if len(bridge_pool) == 0:
                return {
                    **info, "distill_steps": 0, "distill_moved": 0.0,
                    "target": take(anchor, torch.arange(0)), "target_bridge": 0,
                }

            bridge_endpoints = take(bridge_pool, torch.randint(len(bridge_pool), (len(bridge_endpoints),), device=self.device))
            anchor = take(anchor_pool, torch.randint(len(anchor_pool), (len(anchor),), device=self.device))

        target = cat([bridge_endpoints, anchor])
        target_latents = target.to("cpu")

        opt = torch.optim.AdamW(
            self.base_model.parameters(),
            lr=self.config.ft_lr,
            weight_decay=self.config.ft_weight_decay,
        )

        # For the reverse direction, the old model's paths are sampled once and reused for every check
        old_model = copy.deepcopy(self.base_model).eval()
        old_paths = self.sample_paths(old_model, self.base_model) if self.abe.trust_direction == "reverse" else None

        steps = 0
        moved = 0.0
        while steps < self.config.ft_steps:
            prev_state = {k: v.clone() for k, v in self.base_model.state_dict().items()}
            chunk = min(self.abe.distill_check_every, self.config.ft_steps - steps)
            train_base_model(
                self.base_model,
                opt,
                [target_latents],
                steps=chunk,
                batch_size=self.config.ft_batch_size,
                accumulate_steps=self.config.ft_accumulate_steps,
                pbar=False,
            )

            if self.abe.trust_direction == "reverse":
                new_moved = self.path_kl(old_model, self.base_model, old_paths)
            else:
                new_moved = self.path_kl(self.base_model, old_model)
            if new_moved > self.abe.trust_radius:
                info["distill_rejected"] = new_moved
                self.base_model.load_state_dict(prev_state)
                break

            steps += chunk
            moved = new_moved

        return {
            **info,
            "distill_steps": steps,
            "distill_moved": moved,
            "target": target,
            "target_bridge": len(bridge_endpoints),
        }

    def _path_env(self, p: BaseModel[D], q: BaseModel[D]) -> Environment[D]:
        """Environment that samples with p and measures its control relative to q."""
        env = construct_env(q, DummyReward(), self.config.num_steps)
        env.policy = p
        return env

    @torch.no_grad()
    def sample_paths(self, p: BaseModel[D], q: BaseModel[D]) -> Sample[D]:
        """Samples `trust_samples` trajectories of the SDE of model p."""
        return self._path_env(p, q).batch_sample(self.abe.trust_samples, self.config.sample_batch_size)

    @torch.no_grad()
    def path_kl(self, p: BaseModel[D], q: BaseModel[D], paths: Optional[Sample[D]] = None) -> float:
        """Girsanov estimate of KL(P_p || P_q) between the path measures of the SDEs of two models
        with the same noise schedule, an upper bound on the KL between their samples:

            KL(P_p || P_q) = E_{paths ~ p} int_0^1 1/2 ||u_p(x_t, t) - u_q(x_t, t)||^2 dt,

        where u_p - u_q is the difference of their drifts scaled by the noise. Paths of p are sampled unless given, e.g., to reuse them across checks."""
        if paths is None:
            paths = self.sample_paths(p, q)

        env = self._path_env(p, q)
        kwargs = dict_to_device(paths.kwargs, self.device)
        kl = torch.zeros(len(paths), device=self.device)
        for i, (t0, t1) in enumerate(zip(paths.timesteps[:-1], paths.timesteps[1:])):
            x = paths.trajectory[i].to(self.device)
            t = t0 * torch.ones(len(paths), device=self.device)
            _, running_cost = env.drift(x, t, **kwargs)
            kl += (t1 - t0) * running_cost.to(self.device)

        return kl.mean().item()

    def query(self, x: D) -> Batch[D]:
        """Queries the verifier on the selected latents (line 11)."""
        latents = x.to("cpu")
        samples = self.base_model.postprocess(latents.to(self.device)).to("cpu")
        valids = self.problem.validity(samples, {}).cpu().bool()
        return Batch(samples=samples, latents=latents, rewards=torch.zeros(len(x)), valids=valids, kwargs={})

    def explore_loop(self, num_iterations: int, samples_per_iter: Optional[int] = None) -> dict[str, Any]:
        """Runs ABE. `samples_per_iter` is unused, since the verifier is only queried on
        `query_budget` samples per round."""
        cfg = self.abe
        batches: list[Batch[D]] = []
        total_samples = 0
        total_valid_samples = 0
        samples_to_solve = None

        for i in range(num_iterations):
            with self._timer("eval"):
                metrics = {}
                if i % self.config.eval_every == 0:
                    metrics = self.eval_model(i)

            # The surrogate was updated on D_t at the end of the previous round (line 3)
            has_data = len(batches) > 0

            # Disperse under Q from the model and from the frontier of the positives, and tilt the
            # pool towards high acquisition (lines 4-6)
            with self._timer("disperse"):
                source = self.sample_model(cfg.pool_size)
                positives = self.positives(batches) if has_data else take(source, torch.arange(0))

                num_frontier = round(cfg.frontier_frac * cfg.pool_size) if len(positives) > 0 else 0
                starts = take(source, torch.arange(cfg.pool_size - num_frontier))
                if num_frontier > 0:
                    frontier_score, _ = self.acquisition(positives, has_data)
                    frontier_weights, _ = tilt_weights(frontier_score, cfg.tilt, cfg.ess_min)
                    frontier = take(positives, torch.multinomial(frontier_weights, num_frontier, replacement=True))
                    starts = cat([starts, frontier])

                pool = self.disperse(starts, math.sqrt(cfg.step_size * cfg.disperse_time))
                pool_score, pool_feasible = self.acquisition(pool, has_data)
                weights, beta = tilt_weights(pool_score, cfg.tilt, cfg.ess_min, pool_feasible)

            # Build the target from the tilted pool and the positives, and fit the bridge (lines 7-8)
            with self._timer("bridge"):
                target = cat([pool, positives])
                target_weights = torch.cat([len(pool) * weights, torch.ones(len(positives), device=self.device)])
                target_weights = target_weights / target_weights.sum()
                bridge_loss = self.fit_bridge(source, target, target_weights)

            # Propose candidates from the bridge endpoint (line 9)
            with self._timer("propose"):
                candidates = self.propose(cfg.proposal_size, target, target_weights)

            # Select a diverse batch with high acquisition (line 10)
            with self._timer("select"):
                cand_score, cand_feasible = self.acquisition(candidates, has_data)
                if not cand_feasible.any():
                    cand_feasible = torch.ones_like(cand_feasible)

                selected = greedy_diverse(
                    self.features(candidates),
                    cand_score.masked_fill(~cand_feasible, float("-inf")),
                    cfg.query_budget,
                    cfg.exclusion_radius,
                    self.distance,
                )

            # Query verifier on the selected samples only (line 11)
            with self._timer("query"):
                batch = self.query(take(candidates, selected))
                total_samples += len(batch)
                total_valid_samples += batch.valids.int().sum().item()
                solved = self.problem.is_solved(batch)
                if solved and samples_to_solve is None:
                    samples_to_solve = total_samples

            batches.append(batch)

            with self._timer("visualize"):
                if not self.config.no_visualize:
                    self.visualize_iter(batch, i)

                metrics["queried"] = total_samples
                metrics["query_valid"] = batch.valids.float().mean().item()
                metrics["frontier"] = num_frontier
                metrics["pool_feasible"] = pool_feasible.float().mean().item()
                metrics["beta"] = beta
                metrics["ess"] = effective_sample_size(weights)
                metrics["bridge_loss"] = bridge_loss

            # Update the surrogate on D_{t+1} already (line 3 of the next round), such that the
            # distillation filter sees the newest labels
            with self._timer("uncertainty_update"):
                if not self.config.no_uncertainty:
                    self.update_uncertainty_estimator(batches)

            # Distill the bridge into the flow model (line 12)
            with self._timer("finetune"):
                num_bridge = round(cfg.mix_weight * cfg.distill_size)
                bridge_endpoints = self.propose(num_bridge, target, target_weights)
                distilled = self.distill(bridge_endpoints, self.positives(batches))
                metrics.update({k: v for k, v in distilled.items() if k not in ("target", "target_bridge")})

            if cfg.save_rounds:
                # Particles are saved as their distance features, i.e., coordinates for tensor latents
                rounds_dir = self.config.folder / "rounds"
                rounds_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "source": self.features(source).cpu(),
                    "pool": self.features(pool).cpu(),
                    "weights": weights.cpu(),
                    "beta": beta,
                    "target": self.features(target).cpu(),
                    "target_weights": target_weights.cpu(),
                    "candidates": self.features(candidates).cpu(),
                    "selected": selected.cpu(),
                    "valids": batch.valids,
                    "bridge_endpoints": self.features(bridge_endpoints).cpu(),
                    "distill_target": self.features(distilled["target"]).cpu(),
                    "distill_target_bridge": distilled["target_bridge"],
                    "model_samples": self.features(self.sample_model(cfg.distill_size)).cpu(),
                }, rounds_dir / f"{i:04d}.pt")

            metrics["max_vram"] = torch.cuda.max_memory_allocated() * 1e-9
            self.logger.info(f"(iter={i:05d}) {', '.join([f'{k}: {v:.2f}' for k, v in metrics.items()])}")

            with self._timer("checkpoint"):
                self.save_checkpoint(i)

            self._flush_timings(i)

            if solved and self.config.stop_when_solved:
                self.logger.info(f"solved at iteration {i} after {total_samples} queries")
                break

        write_video(
            sorted(self.config.folder.glob("frames/*.png")),
            self.config.folder / "video.mp4",
            fps=self.config.video_fps,
        )

        return {
            "iterations": i + 1,
            "num_samples": total_samples,
            "num_valid_samples": total_valid_samples,
            "solved": samples_to_solve is not None,
            "samples_to_solve": samples_to_solve,
        }


class Empty:
    """Empty set of latents, for latent types that cannot be empty, e.g., batched graphs. `cat` drops
    it, and everything else checks the length before touching latents."""

    def __len__(self) -> int:
        return 0

    def to(self, device: Any) -> "Empty":
        return self


def take(x: D, idx: torch.Tensor) -> D:
    """Selects latents by index, for any latent type."""
    if isinstance(x, DDTensor):
        return DDTensor(x.data[idx.to(x.data.device)])  # type: ignore

    if len(idx) == 0:
        return Empty()  # type: ignore

    return type(x).collate([x[int(i)] for i in idx.tolist()])


def cat(xs: list[D]) -> D:
    """Concatenates latents, for any latent type. Empty sets are dropped unless all are empty."""
    non_empty = [x for x in xs if len(x) > 0]
    if len(non_empty) == 0:
        return xs[0]

    return type(non_empty[0]).collate(non_empty)


def log_sinkhorn(cost: torch.Tensor, reg: float, num_iters: int) -> torch.Tensor:
    """Log of the entropic OT plan between uniform marginals, computed in the log domain."""
    n, m = cost.shape
    f = torch.zeros(n, device=cost.device)
    g = torch.zeros(m, device=cost.device)

    for _ in range(num_iters):
        f = -reg * math.log(n) - reg * torch.logsumexp((g[None, :] - cost) / reg, dim=1)
        g = -reg * math.log(m) - reg * torch.logsumexp((f[:, None] - cost) / reg, dim=0)

    return (f[:, None] + g[None, :] - cost) / reg


def sample_reference(
    n: int,
    d: int,
    s: float,
    cfg: ABEConfig,
    device: torch.device,
    reference: Optional[str] = None,
) -> torch.Tensor:
    """n displacements in d dimensions under the reference Q (`cfg.reference` unless `reference`
    is given) with scale s. The references only differ in their tails:

        gaussian: Brownian motion, s G.
        student:  multivariate Student-t with `reference_df` degrees of freedom, s G / sqrt(chi2 / df).
        levy:     isotropic alpha-stable with characteristic function exp(-|s t|^alpha / 2^(alpha / 2)),
                  i.e., s G at alpha = 2, sampled as sqrt(A) s G with A positive (alpha / 2)-stable.
        poisson:  Brownian motion plus N ~ Poisson(jump_rate tau) Gaussian jumps of std `jump_scale`.

    Displacements longer than `max_jump` are shortened to it."""
    reference = reference or cfg.reference
    g = torch.randn(n, d, device=device)

    if reference == "gaussian":
        disp = s * g
    elif reference == "student":
        chi2 = torch.distributions.Chi2(torch.tensor(float(cfg.reference_df))).sample((n, 1)).to(device)
        disp = s * g / torch.sqrt(chi2 / cfg.reference_df)
    elif reference == "levy":
        disp = s * torch.sqrt(positive_stable(cfg.reference_alpha / 2, n, device)).unsqueeze(1) * g
    elif reference == "poisson":
        num_jumps = torch.poisson(torch.full((n, 1), cfg.jump_rate * cfg.disperse_time, device=device))
        disp = s * g + cfg.jump_scale * torch.sqrt(num_jumps) * torch.randn(n, d, device=device)
    else:
        raise ValueError(f"unknown reference {reference}")

    norm = disp.norm(dim=1, keepdim=True)
    return disp * (cfg.max_jump / norm).clamp(max=1.0)


def positive_stable(a: float, n: int, device: torch.device) -> torch.Tensor:
    """Samples of the positive a-stable law with Laplace transform exp(-s^a), for a in (0, 1], with
    Kanter's representation."""
    if a >= 1:
        return torch.ones(n, device=device)

    u = math.pi * torch.rand(n, device=device).clamp(1e-6, 1 - 1e-6)
    e = torch.empty(n, device=device).exponential_()
    return (torch.sin(a * u) / torch.sin(u) ** (1 / a)) * (torch.sin((1 - a) * u) / e) ** ((1 - a) / a)


def effective_sample_size(weights: torch.Tensor) -> float:
    """ESS of normalized weights as a fraction of the number of particles."""
    return (1.0 / (weights.pow(2).sum() * len(weights))).item()


def tilt_weights(
    score: torch.Tensor,
    beta: float,
    ess_min: float,
    feasible: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, float]:
    """Normalized weights w ~ exp(beta score) on the feasible points, and zero elsewhere. If
    ESS(w) < ess_min, beta is lowered by bisection to the largest value that satisfies the ESS floor
    (line 6), or to zero if even uniform weights on the feasible points do not."""
    if feasible is None or not feasible.any():
        feasible = torch.ones_like(score, dtype=torch.bool)

    def weights_at(b: float) -> torch.Tensor:
        return torch.softmax((b * score).masked_fill(~feasible, float("-inf")), dim=0)

    weights = weights_at(beta)
    if effective_sample_size(weights) >= ess_min:
        return weights, beta

    lo, hi = 0.0, beta
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if effective_sample_size(weights_at(mid)) >= ess_min:
            lo = mid
        else:
            hi = mid

    return weights_at(lo), lo


def greedy_diverse(
    x: torch.Tensor,
    score: torch.Tensor,
    budget: int,
    radius: float,
    distance: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.cdist,
) -> torch.Tensor:
    """Greedily picks up to `budget` points by descending score, skipping any point within `radius`
    of an already picked one under `distance` between the features x. Points with a score of -inf are never picked. Can return fewer than
    `budget` points if the candidates are clustered or infeasible."""
    picked: list[int] = []
    for j in torch.argsort(score, descending=True).tolist():
        if len(picked) == budget or score[j] == float("-inf"):
            break

        if picked and distance(x[j:j+1], x[picked]).min() < radius:
            continue

        picked.append(j)

    return torch.tensor(picked, dtype=torch.long, device=x.device)


def main(args):
    loop = build_loop(
        args,
        reward=DummyReward(),
        mean_weight=0,
        loop_cls=ABELoop,
        abe_config=ABEConfig.construct_from_args(args),
    )
    loop.explore_loop(loop.config.num_iters)


def add_extra_args(parser):
    # Baselines
    parser.add_argument("--no_uncertainty", action="store_true")

    add_abe_args(parser)


def add_abe_args(parser):
    defaults = ABEConfig()
    parser.add_argument("--abe_step_size", type=float, default=defaults.step_size, help="eps")
    parser.add_argument("--abe_disperse_time", type=float, default=defaults.disperse_time, help="tau_t")
    parser.add_argument("--abe_reference", type=str, choices=["gaussian", "student", "levy", "poisson"], default=defaults.reference, help="Q")
    parser.add_argument("--abe_reference_df", type=float, default=defaults.reference_df)
    parser.add_argument("--abe_reference_alpha", type=float, default=defaults.reference_alpha)
    parser.add_argument("--abe_jump_rate", type=float, default=defaults.jump_rate)
    parser.add_argument("--abe_jump_scale", type=float, default=defaults.jump_scale)
    parser.add_argument("--abe_max_jump", type=float, default=defaults.max_jump)
    parser.add_argument("--abe_frontier_frac", type=float, default=defaults.frontier_frac)
    parser.add_argument("--abe_acquisition", type=str, choices=["uncertainty", "ucb"], default=defaults.acquisition)
    parser.add_argument("--abe_feasibility_weight", type=float, default=defaults.feasibility_weight, help="lambda")
    parser.add_argument("--abe_standardize_labels", action="store_true")
    parser.add_argument("--abe_tilt", type=float, default=defaults.tilt, help="beta_t")
    parser.add_argument("--abe_kappa", type=float, default=defaults.kappa)
    parser.add_argument("--abe_ess_min", type=float, default=defaults.ess_min, help="As a fraction of the pool size")
    parser.add_argument("--abe_pool_size", type=int, default=defaults.pool_size, help="N")
    parser.add_argument("--abe_smooth_bandwidth", type=float, default=defaults.smooth_bandwidth)
    parser.add_argument("--abe_bridge", type=str, choices=["cfm", "sb", "none"], default=defaults.bridge)
    parser.add_argument("--abe_bridge_steps", type=int, default=defaults.bridge_steps)
    parser.add_argument("--abe_bridge_batch_size", type=int, default=defaults.bridge_batch_size)
    parser.add_argument("--abe_bridge_lr", type=float, default=defaults.bridge_lr)
    parser.add_argument("--abe_sinkhorn_iters", type=int, default=defaults.sinkhorn_iters)
    parser.add_argument("--abe_proposal_size", type=int, default=defaults.proposal_size, help="B'")
    parser.add_argument("--abe_query_budget", type=int, default=defaults.query_budget, help="B")
    parser.add_argument("--abe_exclusion_radius", type=float, default=defaults.exclusion_radius, help="delta")
    parser.add_argument("--abe_distill_filter", type=str, choices=["lcb", "none"], default=defaults.distill_filter)
    parser.add_argument("--abe_safety_weight", type=float, default=defaults.safety_weight, help="lambda_s")
    parser.add_argument("--abe_mix_weight", type=float, default=defaults.mix_weight, help="alpha")
    parser.add_argument("--abe_trust_radius", type=float, default=defaults.trust_radius, help="rho")
    parser.add_argument("--abe_trust_direction", type=str, choices=["reverse", "forward"], default=defaults.trust_direction)
    parser.add_argument("--abe_distill_size", type=int, default=defaults.distill_size)
    parser.add_argument("--abe_distill_check_every", type=int, default=defaults.distill_check_every)
    parser.add_argument("--abe_trust_samples", type=int, default=defaults.trust_samples)
    parser.add_argument("--abe_save_rounds", action="store_true")


if __name__ == "__main__":
    parser = build_parser(add_extra_args)
    args = parser.parse_args()
    main(args)
