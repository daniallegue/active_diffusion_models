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
from .path_cost import (
    OracleStateCost, bridge_points, path_log_weight, per_step_scale, simulate_path,
    surrogate_state_cost, tilt_weights_with_offset,
)
from .utils import Batch, write_video


# Per-round metrics kept in the summary when the problem defines a shortcut region
HISTORY_KEYS = (
    "mu_band", "mu_band_A", "mu_cross_A", "mu_A", "pool_band", "pool_cross", "pool_left", "band_known",
    "path_alive", "path_cost", "beta", "ess", "ess_alive", "query_valid", "query_A", "query_A_cross",
    "query_A_dirty", "lineage_alive", "repriced_mass",
)


@dataclass(frozen=True)
class ABEConfig:
    # Reference process Q and dispersal
    step_size: float = 0.05          # eps, diffusion coefficient of Q and entropic OT regularization
    disperse_time: float = 1.0       # tau_t, time that the pool is simulated under Q
    reference: Literal["gaussian", "student", "levy", "poisson"] = "gaussian"
    reference_df: float = 2.0        # degrees of freedom of "student"
    reference_alpha: float = 1.5     # stability index of "levy", in (0, 2]
    disperse_steps: int = 1          # increments the dispersal of line 4 is simulated in; 1 is a single displacement

    # State cost V of the reference (Feynman-Kac). Only has an effect with disperse_steps > 1,
    # since a single displacement has no intermediate states to charge.
    state_cost: Literal["none", "surrogate", "oracle"] = "none"  # "oracle" is a diagnostic upper bound
    state_cost_form: Literal["surprise", "excess"] = "surprise"  # -log P(valid), or only its excess over the prior
    state_cost_weight: float = 1.0   # weight on -log P(valid) along the path
    state_cost_hard: bool = False    # also kill a path that crosses mu + lambda sigma < 0
    lineage: bool = False            # re-price each positive's ancestral paths under the current cost
    jump_rate: float = 0.2           # Poisson rate of jumps per unit time for "poisson"
    jump_scale: float = 1.0          # standard deviation of each jump for "poisson"
    max_jump: float = 5.0            # cap on the norm of a dispersal step, for heavy tails

    # Dispersal from the frontier of the verified positives
    frontier_frac: float = 0.5       # fraction of the pool that starts at positives instead of mu_theta

    # Surrogate and tilt
    acquisition: Literal["uncertainty", "ucb", "diversity"] = "uncertainty"
    feasibility_weight: float = 0.25 # lambda, x is feasible iff mu(x) + lambda sigma(x) >= 0
    standardize_labels: bool = False # otherwise, validity labels are centred to +-0.5
    tilt: float = 5.0                # beta_t
    kappa: float = 1.0               # exploration weight of "ucb"
    ess_min: float = 0.1             # floor on ESS(w) / N
    ess_over_alive: bool = False     # ESS as a fraction of the paths left alive, not of N

    # Target and bridge
    pool_size: int = 1024            # N
    positive_mass: Optional[float] = None  # gamma, total mass of D^+ in the target; None gives each positive a pool particle's mass
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
    select_path_weight: bool = False # rank candidates by the target's logit beta alpha + log w, not by alpha alone

    # Distillation
    distill_filter: Literal["lcb", "none"] = "lcb"
    safety_weight: float = 0.1       # lambda_s, x is confidently valid iff mu(x) - lambda_s sigma(x) >= 0
    mix_weight: float = 0.5          # alpha, only used by the "rollback" trust region
    trust: Literal["penalty", "rollback", "none"] = "penalty"  # adaptive Girsanov KL penalty, rollback of chunks past rho, or none
    trust_radius: float = 0.1        # rho
    trust_coef: float = 1.0          # mu, initial weight of the "penalty", adapted to hit rho
    trust_direction: Literal["reverse", "forward"] = "reverse"  # KL(mu_theta || mu_theta') or KL(mu_theta' || mu_theta)
    distill_size: int = 1024
    distill_check_every: int = 50
    trust_samples: int = 256         # paths for the Girsanov estimate of the trust region

    # Diagnostics
    save_rounds: bool = False        # save pool, proposals, selection, and model samples per round
    diagnostic_substeps: int = 8     # Brownian bridge points between recorded states for the crossing diagnostics

    def __post_init__(self):
        if self.positive_mass is not None and not (0 <= self.positive_mass < 1):
            raise ValueError(f"positive_mass must be in [0, 1), got {self.positive_mass}")

        if (self.lineage or self.select_path_weight) and self.bridge != "none":
            raise ValueError(
                "lineage and select_path_weight need each candidate's pool path, which only "
                "bridge='none' keeps: a fitted bridge proposes from fresh model samples"
            )

        if not (0 < self.step_size < 1):
            raise ValueError(f"step_size must be in (0, 1), got {self.step_size}")

        if not (0 <= self.mix_weight <= 1):
            raise ValueError(f"mix_weight must be in [0, 1], got {self.mix_weight}")

        if self.trust == "penalty" and self.trust_direction != "reverse":
            raise ValueError("the penalty trust region requires trust_direction='reverse'")

        if self.disperse_steps < 1:
            raise ValueError(f"disperse_steps must be at least 1, got {self.disperse_steps}")

        if self.state_cost != "none" and self.disperse_steps == 1:
            raise ValueError(
                "a state cost needs disperse_steps > 1: a single displacement has no intermediate "
                "states to charge, so V would only ever be evaluated at the pool's endpoints, which "
                "the feasibility mask of line 5 already does"
            )

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
        self._positive_feats: Optional[torch.Tensor] = None

        # Per positive, in the order of `positives`: the positive whose frontier particle found it
        # (-1 for a model sample), and the recorded paths, as (indices, states) chunks
        self._lineage_parent: list[int] = []
        self._lineage_paths: list[tuple[torch.Tensor, list]] = []

        # Oracle crossing diagnostics, on problems that define a shortcut region. Positives are
        # "clean" if neither their own path nor any ancestor's passed through it.
        self._diagnose = isinstance(x, DDTensor) and hasattr(self.problem, "in_shortcut")
        self._diag_gen = torch.Generator().manual_seed(torch.initial_seed() % 2**63)
        self._pos_clean: list[bool] = []
        self.trust_coef = self.abe.trust_coef
        if isinstance(x, DDTensor):
            self.latent_shape = x.data.shape[1:]

        if self.abe.bridge != "none":
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

    def disperse(self, x: D, scale: float, reference: Optional[str] = None, cfg: Optional[ABEConfig] = None) -> D:
        """Disperses latents under the reference Q with the given scale (line 4). The problem setup
        decides how a displacement acts on its latents.

        `cfg` overrides the reference parameters for this increment, which `disperse_path` uses to
        thin the Poisson rate when it splits a dispersal into steps."""
        if len(x) == 0 or scale <= 0:
            return x

        ref_cfg = cfg if cfg is not None else self.abe

        def noise(n: int, d: int) -> torch.Tensor:
            return sample_reference(n, d, scale, ref_cfg, self.device, reference)

        return self.problem.disperse(self.env, x, scale, noise)

    def state_cost(self, has_data: bool) -> Optional[Any]:
        """The state cost V of line 4, or None when the reference carries no cost. Needs a fitted
        surrogate, so the first rounds disperse under the plain reference."""
        if self.abe.state_cost == "oracle":
            # Known from the first round, which is the point of it: see OracleStateCost
            return OracleStateCost(
                lambda x: self.problem.validity(self.base_model.postprocess(x), {}),
                self.abe.state_cost_weight,
                self.abe.state_cost_hard,
                self.device,
            )

        if self.abe.state_cost == "none" or not has_data or self.config.no_uncertainty:
            return None

        if not self._surrogate_fitted:
            return None

        return surrogate_state_cost(
            self.uncertainty.mean_and_uncertainty,
            self.abe.state_cost_weight,
            self.abe.state_cost_hard,
            self.abe.feasibility_weight,
            self.device,
            form=self.abe.state_cost_form,
        )

    def disperse_path(
        self, x: D, scale: float, has_data: bool, monitor: Optional[Callable[[int, D], None]] = None,
    ) -> tuple[D, torch.Tensor, dict[str, float]]:
        """Simulates the dispersal of line 4 as a path and returns the endpoint together with the
        Feynman-Kac log weight exp(-int V) that turns the pool into a sample of rho_t K^V_tau.

        With `disperse_steps = 1` and no state cost this is the single displacement of the original
        line 4 and the weights are zero, so the default path through the loop is unchanged."""
        return simulate_path(
            x,
            scale,
            self.abe.disperse_steps,
            lambda y, s, c: self.disperse(y, s, cfg=c),
            self.abe,
            self.state_cost(has_data),
            self.device,
            monitor,
        )

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

        loose = mean + self.abe.feasibility_weight * uncert >= 0
        if self.abe.acquisition == "diversity":
            # Distance to the nearest verified positive, so the tilt rewards leaving D^+ rather than
            # being uncertain about it. Falls back to uncertainty until there are positives to be
            # far from.
            feats = self._positive_feats
            if feats is not None and len(feats) > 0:
                return self.distance(self.features(x), feats).min(dim=1).values, loose

        return uncert, loose

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
        return self.propose_indexed(n, target, target_weights)[0]

    def propose_indexed(self, n: int, target: D, target_weights: torch.Tensor) -> tuple[D, torch.Tensor]:
        """`propose`, together with the index into `target` that each proposal was resampled from,
        or -1 when a bridge proposes from fresh model samples and there is no such particle."""
        if self.abe.bridge != "none":
            return self.simulate_bridge(self.sample_model(n)), torch.full((n,), -1, device=self.device)  # type: ignore

        idx = torch.multinomial(target_weights, n, replacement=True)
        return self.disperse(take(target, idx), self.abe.smooth_bandwidth, reference="gaussian"), idx

    def mix_target(self, weights: torch.Tensor, num_positives: int, positive_log_w: Optional[torch.Tensor]) -> torch.Tensor:
        """Weights of the target `cat([pool, positives])` (line 7).

        With `positive_mass` unset each positive weighs as much as a pool particle, so D^+ takes a
        share |D^+| / (N + |D^+|) of the target that grows every round, and its points carry neither
        the tilt nor a path weight. With gamma set the pool keeps 1 - gamma whatever |D^+| is, and
        gamma = 0 leaves exactly mu*, the frontier already dispersing from D^+. The positives' share
        is spread by their lineage weight, uniformly without one.
        """
        cfg = self.abe
        if cfg.positive_mass is None:
            target_weights = torch.cat([len(weights) * weights, torch.ones(num_positives, device=self.device)])
            return target_weights / target_weights.sum()

        gamma = cfg.positive_mass if num_positives > 0 else 0.0
        pos = torch.full((num_positives,), 1.0 / max(1, num_positives), device=self.device)
        if positive_log_w is not None and positive_log_w.isfinite().any():
            pos = torch.softmax(positive_log_w, dim=0)

        return torch.cat([(1 - gamma) * weights, gamma * pos])

    def lineage_log_weight(self, has_data: bool) -> Optional[torch.Tensor]:
        """Log weight of every positive's whole ancestry under the *current* cost, or None.

        A positive found by a frontier particle inherits its parent's weight, so this is the
        Feynman-Kac weight of the concatenated path back to a model sample. That is what makes
        "reached through valid regions" hold across rounds and not only within one: a target found
        in round 1 by crossing a wall nobody had measured yet is pruned the moment the surrogate
        learns the wall, instead of serving as a clean frontier start for ever after.

        Model samples start a lineage clean, since the flow model keeps no record of how its mass
        got where it is. That is the remaining leak, and why distillation filters on lineage too.
        """
        if not self.abe.lineage or len(self._lineage_parent) == 0:
            return None

        n = len(self._lineage_parent)
        own = torch.zeros(n, device=self.device)
        cost = self.state_cost(has_data)
        if cost is not None:
            for idx, states in self._lineage_paths:
                own[idx] = path_log_weight(states, cost, self.device)

        total = own.tolist()
        for j, parent in enumerate(self._lineage_parent):
            if parent >= 0:
                total[j] += total[parent]

        return torch.tensor(total, device=self.device)

    @torch.no_grad()
    def crossings(self, states: list, step_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Oracle diagnostics of recorded paths: whether each passes through the problem's shortcut
        region, and whether it leaves the valid set at all. Measured on the recorded states and on
        `diagnostic_substeps` Brownian bridge points between each pair, so that a crossing between
        two of the cost's own states is still seen. Uses a private generator: observing a run must
        not change it."""
        m = self.abe.diagnostic_substeps if self.abe.reference == "gaussian" else 0
        pts = [self._flat(x) for x in states]
        checks = list(pts)
        for a, b in zip(pts[:-1], pts[1:]):
            if m > 0:
                checks.extend(bridge_points(a, b, step_scale, m, self._diag_gen).unbind(0))

        band = torch.zeros(len(pts[0]), dtype=torch.bool, device=self.device)
        left = torch.zeros_like(band)
        for x in checks:
            samples = self.base_model.postprocess(self._unflat(x))
            band |= self.problem.in_shortcut(samples).to(self.device)
            left |= ~self.problem.validity(samples, {}).to(self.device).bool()

        return band, left

    def distill(self, bridge_endpoints: D, positives: D) -> dict[str, Any]:
        """Distills the bridge endpoints into the flow model for `ft_steps` within a trust region of
        radius rho on the Girsanov KL between the path measures of the old and new model (`path_kl`),
        by default KL(P_old || P_new), which only penalizes losing mass that the old model had."""
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

        old_model = copy.deepcopy(self.base_model).eval().requires_grad_(False)
        if self.abe.trust in ("penalty", "none"):
            return {
                **info,
                **self.distill_penalized(target_latents, old_model),
                "target": target,
                "target_bridge": len(bridge_endpoints),
            }

        # For the reverse direction, the old model's paths are sampled once and reused for every check
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

    def distill_penalized(self, target: D, old_model: BaseModel[D]) -> dict[str, Any]:
        """Trains the flow model on `target` for `ft_steps` with the loss

            L = L_FM(target) + mu KL(P_old || P_new),

        where the Girsanov KL is estimated on states of `trust_samples` paths of the old model.
        """
        if self.abe.trust == "none":
            self.trust_coef = 0.0

        rho = self.abe.trust_radius
        bs = self.config.ft_batch_size
        accumulate_steps = self.config.ft_accumulate_steps

        paths = self.sample_paths(old_model, self.base_model)
        env = self._path_env(old_model, self.base_model)
        span = (paths.timesteps[-1] - paths.timesteps[0]).item()

        opt = torch.optim.AdamW(
            self.base_model.parameters(),
            lr=self.config.ft_lr,
            weight_decay=self.config.ft_weight_decay,
        )

        fm_sum = 0.0
        penalty_sum = 0.0
        kl = 0.0
        self.base_model.train()
        opt.zero_grad()
        for step in range(1, self.config.ft_steps + 1):
            x1 = take(target, torch.randint(len(target), (bs,))).to(self.device)
            fm = self.base_model.train_loss(x1).mean()

            # int_0^1 L(x_t, t) dt is span times the mean running cost at a uniform grid time. 
            penalty = torch.zeros((), device=self.device)
            if self.abe.trust == "penalty":
                i = int(torch.randint(len(paths.timesteps) - 1, ()))
                xt = take(paths.trajectory[i], torch.randint(len(paths), (bs,))).to(self.device)
                t = paths.timesteps[i].item() * torch.ones(bs, device=self.device)
                _, running_cost = env.drift(xt, t)
                penalty = span * running_cost.mean()

            loss = fm + self.trust_coef * penalty
            (loss / accumulate_steps).backward()
            if step % accumulate_steps == 0:
                # Same clipping as `train_base_model`
                torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), 0.1)
                opt.step()
                opt.zero_grad()

            fm_sum += fm.item()
            penalty_sum += penalty.item()

            if step % self.abe.distill_check_every == 0 or step == self.config.ft_steps:
                self.base_model.eval()
                kl = self.path_kl(old_model, self.base_model, paths)
                self.base_model.train()

                if self.abe.trust != "penalty":
                    continue

                if kl > 1.5 * rho:
                    self.trust_coef *= 2
                elif kl < rho / 1.5:
                    self.trust_coef /= 2

                # Keeps mu from running away while the model does not move, e.g., without data
                self.trust_coef = min(max(self.trust_coef, 1e-4), 1e4)

        self.base_model.eval()
        steps = max(1, self.config.ft_steps)
        return {
            "distill_steps": self.config.ft_steps,
            "distill_moved": kl,
            "distill_fm": fm_sum / steps,
            "distill_penalty": penalty_sum / steps,
            "trust_coef": self.trust_coef,
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
        samples_to_solve_clean = None
        totals = {"query_A": 0, "query_A_cross": 0, "query_A_dirty": 0}
        history: list[dict[str, float]] = []

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
                lineage_log_w = self.lineage_log_weight(has_data)

                num_frontier = round(cfg.frontier_frac * cfg.pool_size) if len(positives) > 0 else 0
                starts = take(source, torch.arange(cfg.pool_size - num_frontier))
                start_parent = torch.full((cfg.pool_size - num_frontier,), -1, dtype=torch.long)
                if num_frontier > 0:
                    frontier_score, _ = self.acquisition(positives, has_data)
                    if lineage_log_w is None:
                        frontier_weights, _ = tilt_weights(frontier_score, cfg.tilt, cfg.ess_min)
                    else:
                        # Resampling the frontier by lineage weight is the resampling step of the
                        # particle system, so the pool it seeds needs no further correction
                        frontier_weights, _ = tilt_weights_with_offset(
                            frontier_score, cfg.tilt, cfg.ess_min, lineage_log_w
                        )
                    frontier_idx = torch.multinomial(frontier_weights, num_frontier, replacement=True)
                    starts = cat([starts, take(positives, frontier_idx)])
                    start_parent = torch.cat([start_parent, frontier_idx.cpu()])

                self._positive_feats = self.features(positives) if len(positives) > 0 else None
                states: list = []
                scale = math.sqrt(cfg.step_size * cfg.disperse_time)
                pool, path_log_w, path_info = self.disperse_path(
                    starts, scale, has_data,
                    (lambda k, x: states.append(x)) if (self._diagnose or cfg.lineage) else None,
                )
                pool_score, pool_feasible = self.acquisition(pool, has_data)
                weights, beta = tilt_weights_with_offset(
                    pool_score, cfg.tilt, cfg.ess_min, path_log_w, pool_feasible, cfg.ess_over_alive
                )

                pool_cross = None
                if self._diagnose:
                    step_scale = per_step_scale(scale, cfg.disperse_steps, cfg.reference, cfg.reference_alpha)
                    pool_band, pool_left = self.crossings(states, step_scale)
                    # A crossing joins the two sides: through the band, with one end in A and the
                    # other not. A path that leaves A, dips into the band below it and returns is an
                    # excursion like any other, not a shortcut.
                    side = [self.problem.in_target(self.base_model.postprocess(s)).to(self.device) for s in (states[0], states[-1])]
                    pool_cross = pool_band & (side[0] != side[1])
                    metrics.update(self.pool_diagnostics(pool, weights, pool_band, pool_cross, pool_left))

            # Build the target from the tilted pool and the positives, and fit the bridge (lines 7-8)
            with self._timer("bridge"):
                target = cat([pool, positives])
                target_weights = self.mix_target(weights, len(positives), lineage_log_w)
                target_offset = torch.cat([
                    path_log_w,
                    lineage_log_w if lineage_log_w is not None else torch.zeros(len(positives), device=self.device),
                ])
                bridge_loss = self.fit_bridge(source, target, target_weights)

            # Propose candidates from the bridge endpoint (line 9)
            with self._timer("propose"):
                candidates, cand_idx = self.propose_indexed(cfg.proposal_size, target, target_weights)

            # Select a diverse batch with high acquisition (line 10)
            with self._timer("select"):
                cand_score, cand_feasible = self.acquisition(candidates, has_data)
                if not cand_feasible.any():
                    cand_feasible = torch.ones_like(cand_feasible)

                # Ranking by alpha alone discards the path weight: a candidate is only ever in or
                # out of the resample. The target's own logit beta alpha + log w keeps it.
                rank = cand_score
                if cfg.select_path_weight:
                    rank = beta * cand_score + target_offset[cand_idx]

                selected = greedy_diverse(
                    self.features(candidates),
                    rank.masked_fill(~cand_feasible, float("-inf")),
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

                query_diag = self.record_positives(
                    batch, cand_idx[selected], len(pool), start_parent, states, pool_cross, positives
                )
                if self._diagnose:
                    metrics.update(query_diag)
                    for k in ("query_A", "query_A_cross", "query_A_dirty"):
                        totals[k] += query_diag[k]
                    if query_diag["query_A"] > query_diag["query_A_dirty"] and samples_to_solve_clean is None:
                        samples_to_solve_clean = total_samples

            batches.append(batch)

            with self._timer("visualize"):
                if not self.config.no_visualize:
                    self.visualize_iter(batch, i)

                metrics["queried"] = total_samples
                metrics["query_valid"] = batch.valids.float().mean().item()
                metrics["frontier"] = num_frontier
                metrics["pool_feasible"] = pool_feasible.float().mean().item()
                metrics["beta"] = beta
                metrics.update(path_info)
                metrics["ess"] = effective_sample_size(weights)
                metrics["ess_alive"] = metrics["ess"] * len(weights) / max(1, int(path_log_w.isfinite().sum()))
                metrics["bridge_loss"] = bridge_loss

            # Update the surrogate on D_{t+1} already (line 3 of the next round), such that the
            # distillation filter sees the newest labels
            with self._timer("uncertainty_update"):
                if not self.config.no_uncertainty:
                    self.update_uncertainty_estimator(batches)

            # Distill the bridge into the flow model (line 12)
            with self._timer("finetune"):
                num_bridge = round(cfg.mix_weight * cfg.distill_size) if cfg.trust == "rollback" else cfg.distill_size
                distill_positives = self.positives(batches)
                distill_weights = target_weights
                if cfg.lineage:
                    distill_weights, distill_positives, lineage_info = self.reprice_for_distill(
                        weights, path_log_w, states, positives, distill_positives
                    )
                    metrics.update(lineage_info)

                bridge_endpoints = self.propose(num_bridge, target, distill_weights)
                distilled = self.distill(bridge_endpoints, distill_positives)
                metrics.update({k: v for k, v in distilled.items() if k not in ("target", "target_bridge")})
                if self._diagnose:
                    history.append({k: metrics[k] for k in HISTORY_KEYS if k in metrics})

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

        summary = {
            "iterations": i + 1,
            "num_samples": total_samples,
            "num_valid_samples": total_valid_samples,
            "solved": samples_to_solve is not None,
            "samples_to_solve": samples_to_solve,
        }
        if self._diagnose:
            summary.update({
                "samples_to_solve_clean": samples_to_solve_clean,
                "positives_A": totals["query_A"],
                "positives_A_cross": totals["query_A_cross"],
                "positives_A_dirty": totals["query_A_dirty"],
                "rounds": history,
            })

        return summary

    def pool_diagnostics(
        self, pool: D, weights: torch.Tensor, band: torch.Tensor, cross: torch.Tensor, left: torch.Tensor,
    ) -> dict[str, float]:
        """How much of mu* the pool's paths carry across the shortcut region, measured by the
        oracle. `mu_cross_A` is the quantity a path cost exists to drive down: target mass on valid
        endpoints in A whose path got there by crossing from the other side. `band_known` is the
        share of the shortcut region the surrogate's own hard rule already calls invalid, i.e. how
        much of the wall a surrogate cost can see at all."""
        samples = self.base_model.postprocess(pool)
        in_A = self.problem.in_target(samples).to(self.device) & self.problem.validity(samples, {}).to(self.device).bool()
        out = {
            "pool_left": left.float().mean().item(),
            "pool_band": band.float().mean().item(),
            "pool_cross": cross.float().mean().item(),
            "mu_band": weights[band].sum().item(),
            "mu_A": weights[in_A].sum().item(),
            "mu_band_A": weights[band & in_A].sum().item(),
            "mu_cross_A": weights[cross & in_A].sum().item(),
        }

        box = getattr(self.problem, "shortcut_box", None)
        if box is not None and self._surrogate_fitted and not self.config.no_uncertainty:
            xmin, xmax, ymin, ymax = box
            gx, gy = torch.meshgrid(torch.linspace(xmin, xmax, 21), torch.linspace(ymin, ymax, 7), indexing="ij")
            grid = DDTensor(torch.stack([gx.flatten(), gy.flatten()], dim=1).reshape(-1, *self.latent_shape)).to(self.device)
            mean, unc = self.uncertainty.mean_and_uncertainty(grid)
            out["band_known"] = (mean + self.abe.feasibility_weight * unc < 0).float().mean().item()

        return out

    def record_positives(
        self, batch: Batch[D], target_idx: torch.Tensor, num_pool: int, start_parent: torch.Tensor,
        states: list, pool_cross: Optional[torch.Tensor], positives: D,
    ) -> dict[str, float]:
        """Appends the lineage of this round's positives: the positive whose frontier particle found
        each (-1 for a model sample), with `lineage` its path, and the oracle's verdict on whether it
        or any ancestor crossed. A candidate resampled from D^+ itself has a jitter of bandwidth h
        for a path and the positive it came from as its parent."""
        valid = batch.valids.to(self.device)
        new = valid.nonzero().squeeze(1)
        tidx = target_idx[new].tolist()
        first = len(self._lineage_parent)

        parents, own_cross = [], []
        for t in tidx:
            if 0 <= t < num_pool:
                parents.append(int(start_parent[t]))
                own_cross.append(bool(pool_cross[t]) if pool_cross is not None else False)
            else:
                parents.append(t - num_pool if t >= num_pool else -1)
                own_cross.append(False)

        clean = [not b and (p < 0 or self._pos_clean[p]) for p, b in zip(parents, own_cross)]
        self._lineage_parent.extend(parents)
        self._pos_clean.extend(clean)

        if self.abe.lineage and len(new) > 0:
            t = torch.tensor(tidx, device=self.device)
            gidx = torch.arange(first, first + len(new), device=self.device)
            from_pool = (t >= 0) & (t < num_pool)
            if from_pool.any():
                self._lineage_paths.append((gidx[from_pool], [take(s, t[from_pool]) for s in states]))

            from_pos = t >= num_pool
            if from_pos.any():
                ends = take(batch.latents.to(self.device), new[from_pos])
                self._lineage_paths.append((gidx[from_pos], [take(positives, t[from_pos] - num_pool), ends]))

        if not self._diagnose:
            return {}

        in_A = self.problem.in_target(batch.samples).to(self.device)[new].tolist()
        return {
            "query_A": sum(in_A),
            "query_A_cross": sum(a and b for a, b in zip(in_A, own_cross)),
            "query_A_dirty": sum(a and not c for a, c in zip(in_A, clean)),
        }

    def reprice_for_distill(
        self, weights: torch.Tensor, path_log_w: torch.Tensor, states: list, positives: D, all_positives: D,
    ) -> tuple[torch.Tensor, D, dict[str, float]]:
        """Re-prices the round's target under the surrogate as updated by this round's labels, for
        distillation (line 12). The pool was weighted with the cost of the round's *start*, so a
        wall learnt from this very batch would otherwise only take effect a round late -- by which
        time the model has been trained on what crossed it. Positives whose lineage is now dead are
        dropped from the distillation set."""
        cost = self.state_cost(True)
        lineage = self.lineage_log_weight(True)

        new_weights = weights
        if cost is not None and len(states) > 0:
            alive = weights > 0
            logits = torch.full_like(weights, float("-inf"))
            logits[alive] = weights[alive].log() - path_log_w[alive] + path_log_weight(states, cost, self.device)[alive]
            if logits.isfinite().any():
                new_weights = torch.softmax(logits, dim=0)

        lineage_pos = lineage[: len(positives)] if lineage is not None else None
        distill_weights = self.mix_target(new_weights, len(positives), lineage_pos)

        info = {"repriced_mass": (new_weights - weights).abs().sum().item() / 2}
        if lineage is not None and len(lineage) > 0:
            keep = lineage.isfinite()
            info["lineage_alive"] = keep.float().mean().item()
            all_positives = take(all_positives, keep.nonzero().squeeze(1))

        return distill_weights, all_positives, info


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
    parser.add_argument("--abe_disperse_steps", type=int, default=defaults.disperse_steps,
                        help="Increments the dispersal of line 4 is simulated in; >1 is needed for any state cost")
    parser.add_argument("--abe_state_cost", type=str, choices=["none", "surrogate", "oracle"], default=defaults.state_cost,
                        help="V in the Feynman-Kac reference: 'surrogate' charges -log P(valid) along the path, "
                             "'oracle' reads true validity (a diagnostic: its calls are not budgeted)")
    parser.add_argument("--abe_state_cost_form", type=str, choices=["surprise", "excess"], default=defaults.state_cost_form,
                        help="'excess' charges only -log P(valid) beyond the prior's log 2, so unknown space is free")
    parser.add_argument("--abe_state_cost_weight", type=float, default=defaults.state_cost_weight)
    parser.add_argument("--abe_state_cost_hard", action="store_true",
                        help="Also kill a path that crosses mu + lambda sigma < 0, i.e. prune it")
    parser.add_argument("--abe_lineage", action="store_true",
                        help="Re-price every positive's ancestral paths under the current cost, for the frontier, target and distillation")
    parser.add_argument("--abe_ess_over_alive", action="store_true",
                        help="Measure the ESS floor against the paths left alive rather than the whole pool")
    parser.add_argument("--abe_positive_mass", type=float, default=defaults.positive_mass,
                        help="gamma, total mass of the positives in the target; unset gives each one a pool particle's mass")
    parser.add_argument("--abe_select_path_weight", action="store_true",
                        help="Rank candidates by beta alpha + log w, so the path weight reaches the queries")
    parser.add_argument("--abe_diagnostic_substeps", type=int, default=defaults.diagnostic_substeps)
    parser.add_argument("--abe_acquisition", type=str, choices=["uncertainty", "ucb", "diversity"], default=defaults.acquisition)
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
    parser.add_argument("--abe_trust", type=str, choices=["penalty", "rollback", "none"], default=defaults.trust)
    parser.add_argument("--abe_trust_radius", type=float, default=defaults.trust_radius, help="rho")
    parser.add_argument("--abe_trust_coef", type=float, default=defaults.trust_coef, help="mu")
    parser.add_argument("--abe_trust_direction", type=str, choices=["reverse", "forward"], default=defaults.trust_direction)
    parser.add_argument("--abe_distill_size", type=int, default=defaults.distill_size)
    parser.add_argument("--abe_distill_check_every", type=int, default=defaults.distill_check_every)
    parser.add_argument("--abe_trust_samples", type=int, default=defaults.trust_samples)
    parser.add_argument("--abe_save_rounds", action="store_true")


if __name__ == "__main__":
    parser = build_parser(add_extra_args)
    args = parser.parse_args()
    main(args)
