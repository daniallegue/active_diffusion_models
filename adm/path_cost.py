"""Feynman-Kac path costs for ABE's reference process.

ABE's target is the degenerate Schrodinger bridge: setting rho_1 := rho_t K_tau makes the bridge
from rho_t to rho_1 with reference K_tau equal to the reference itself, so KL(P*||Q) = 0, nothing is
estimated, and the tilt e^{beta alpha} is an h transform of K_tau that is exact on the pool. The
target of line 5 is then

    mu*_t  ∝  (rho_t K_tau) e^{beta alpha_t} 1_{F_t}.

A state cost V adds the only thing an endpoint tilt cannot express. By Feynman-Kac the reference
absorbs it,

    dK^V_tau / dK_tau (path)  =  exp(-∫_0^tau V(X_s) ds),

and the target becomes mu*_t ∝ (rho_t K^V_tau) e^{beta alpha_t} 1_{F_t}. This is motivation (3): the
reachability factor, which favours endpoints reached *through* valid regions. It is not a function
of the endpoint, so no choice of terminal reward reproduces it.

**The implementation trap.** It is tempting to absorb V by renormalising the reference's transitions
locally, to something proportional to Q(dy|x) e^{-V(x) dt}. That is a different process. The true
Feynman-Kac law is the h transform

    K^V(dy|x)  ∝  Q(dy|x) e^{-V(x) dt} h(y),      h(x) = E_Q[e^{-∫ V} | X_0 = x],

and dropping h(y) makes the walk myopic: it avoids cost underfoot but is blind to cost ahead.
Computing h means solving a backward problem, which is what the degenerate construction exists to
avoid.

The way out is that ABE never needs the normalised process. It needs *samples weighted toward* the
target, and lines 5 and 6 already perform self-normalised importance sampling on the pool. So the
particles are simulated under the plain reference Q and carry

    log w_i  =  -∫_0^tau V(X^i_s) ds,

which is an exact, h-free Radon-Nikodym derivative of K^V against K. The weight costs one surrogate
call per step per particle and no backward solve. Hard constraints are the V = infinity case: the
particle is killed, its weight is zero, and the path is pruned.
"""

from typing import Callable, Optional, Protocol
from dataclasses import replace
import math
import warnings

import torch


class Disperser(Protocol):
    """One increment of the reference process: a displacement of the given scale under `cfg`.

    `cfg` is passed per step rather than captured, because the Poisson reference's jump rate must be
    thinned when a dispersal is split, and `simulate_path` is the only place that knows by how much.
    Letting the caller bind the config instead makes that correction easy to forget, and forgetting
    it is silent: the endpoint law gains a factor of `num_steps` more jumps.
    """

    def __call__(self, x, scale: float, cfg): ...


def thinned_config(cfg, num_steps: int):
    """The reference config for one increment of a dispersal split into `num_steps`.

    Only the Poisson reference needs it: its jump count is drawn as Poisson(jump_rate *
    disperse_time), so the rate is thinned per step and the sum of the increments' counts is again
    Poisson(jump_rate * disperse_time). The other references carry their whole scale in `scale`.
    """
    if num_steps <= 1 or cfg.reference != "poisson":
        return cfg

    return replace(cfg, disperse_time=cfg.disperse_time / num_steps)


def per_step_scale(scale: float, num_steps: int, reference: str, alpha: float) -> float:
    """The per-increment scale that leaves the endpoint law of the reference unchanged when a
    dispersal of total `scale` is split into `num_steps` increments.

    Gaussian and the alpha-stable Levy reference are stable laws, so splitting is exact at
    s / n^{1/alpha} (alpha = 2 recovers the Gaussian's s / sqrt(n)). The Poisson reference splits
    exactly too, once its rate is thinned by `thinned_config`.

    The Student-t reference is *not* stable, so splitting genuinely distorts its endpoint law.
    Measured as the worst relative gap between matched quantiles against a single displacement
    (tests/test_path_cost.py):

        reference        n=2     n=8    n=64
        gaussian       0.015   0.006   0.015
        levy a=1.5     0.012   0.014   0.008
        poisson        0.038   (thinned)
        student df=2   0.163   0.369   0.521
        student df=5   0.086   0.148   0.186
        student df=30  0.033   0.026   0.031

    The distortion is in the body of the law, not the tail: at df = 2 the 0.999 quantile is
    essentially unchanged. So a split Student-t keeps the heavy tail that Section 8 of the corridor
    report found decisive for disconnected islands, while shifting its bulk. `simulate_path` warns
    rather than refuses, because refusing would cost that tail for a distortion the caller may well
    accept at small `num_steps`.
    """
    if num_steps <= 1:
        return scale

    if reference == "levy":
        return scale / num_steps ** (1.0 / max(alpha, 1e-3))

    return scale / math.sqrt(num_steps)


def _charge(state_cost, state, k: int, num_steps: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    """V, the alive mask and the trapezoid factor of state k of a path of `num_steps` increments.

    The trapezoid rule counts the two endpoints half. A cost with `interior_only` charges and kills
    at the intermediate states only, so that it measures the path and leaves the endpoints to the
    feasibility mask of line 5.
    """
    v, ok = state_cost(state)
    v = v.clamp_min(0.0)
    endpoint = k == 0 or k == num_steps
    if endpoint and getattr(state_cost, "interior_only", False):
        return torch.zeros_like(v), torch.ones_like(ok), 0.0

    return v, ok, 0.5 if endpoint else 1.0


@torch.no_grad()
def path_log_weight(
    states: list,
    state_cost: Callable[[object], tuple[torch.Tensor, torch.Tensor]],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """The Feynman-Kac log weight -int V of recorded paths, under a possibly newer `state_cost`.

    `states[k]` holds state k of every path, as recorded by the `monitor` of `simulate_path`, and
    this applies exactly the rule `simulate_path` does, so on the cost a path was simulated under it
    returns that path's `log_w`. Its use is to re-price old paths once the surrogate has learnt
    more, which `simulate_path` cannot: it only sees the cost of the round it runs in.
    """
    num_steps = len(states) - 1
    n = len(states[0]) if states else 0
    log_w = torch.zeros(n, device=device)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    if n == 0 or num_steps < 1:
        return log_w

    dt = 1.0 / num_steps
    for k, state in enumerate(states):
        v, ok, factor = _charge(state_cost, state, k, num_steps)
        alive &= ok.to(alive.device)
        log_w = log_w - factor * dt * v.to(log_w.device)

    return log_w.masked_fill(~alive, float("-inf"))


@torch.no_grad()
def bridge_points(xa: torch.Tensor, xb: torch.Tensor, s: float, m: int, generator: torch.Generator) -> torch.Tensor:
    """m interior points of a Brownian bridge of per-increment scale s from xa to xb, of shape
    (m, n, d), drawn from `generator` so that a diagnostic does not move the run's RNG.

    A path recorded at `disperse_steps` states can step over a thin wall between two of them, which
    is exactly the resolution limit of the cost. The bridge fills the gap in law, so a crossing
    measured on it is the continuous path's and not the cost's own discretisation of it.
    """
    u = torch.arange(1, m + 1, device=xa.device, dtype=xa.dtype)[:, None, None] / (m + 1)
    noise = torch.randn((m, *xa.shape), generator=generator, dtype=xa.dtype).to(xa.device)
    return (1 - u) * xa + u * xb + s * torch.sqrt(u * (1 - u)) * noise


@torch.no_grad()
def simulate_path(
    x0,
    scale: float,
    num_steps: int,
    disperse: Disperser,
    cfg,
    state_cost: Optional[Callable[[object], tuple[torch.Tensor, torch.Tensor]]] = None,
    device: Optional[torch.device] = None,
    monitor: Optional[Callable[[int, object], None]] = None,
) -> tuple[object, torch.Tensor, dict[str, float]]:
    """Simulates the reference from `x0` for `num_steps` increments and accumulates the Feynman-Kac
    log weight along the way.

    `monitor(k, state)` is called on every state k = 0..num_steps, e.g., to record the path for
    `path_log_weight` or to measure it against an oracle. It must not draw from the global RNG, so
    that observing a run does not change it.

    Returns the endpoint, `log_w` of shape (n,), and diagnostics. With `num_steps == 1` and no state
    cost this is exactly the single displacement of line 4, so the default path through the loop is
    unchanged.

    `state_cost` returns `(V, alive)` at a state: a non-negative cost per particle and a mask that is
    False where the constraint is violated. A killed particle keeps its position, so that the batch
    stays rectangular, but its weight is -inf and nothing downstream can select it.
    """
    n = len(x0)
    if n == 0 or scale <= 0 or num_steps < 1:
        return x0, torch.zeros(n, device=device), {}

    reference = cfg.reference
    if num_steps > 1 and reference == "student":
        warnings.warn(
            f"the Student-t reference is not a stable law, so splitting dispersal into {num_steps} "
            "steps shifts the bulk of its endpoint law (the tail survives); see per_step_scale for "
            "the measured distortion",
            RuntimeWarning,
            stacklevel=2,
        )

    step_cfg = thinned_config(cfg, num_steps)
    s = per_step_scale(scale, num_steps, reference, cfg.reference_alpha)
    dt = 1.0 / num_steps  # tau is carried by `scale`, so the integral is over the unit interval

    x = x0
    log_w = torch.zeros(n, device=device)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    cost_sum = torch.zeros(n, device=device)

    def accumulate(state, k: int):
        nonlocal log_w, alive, cost_sum
        if monitor is not None:
            monitor(k, state)

        if state_cost is None:
            return

        v, ok, factor = _charge(state_cost, state, k, num_steps)
        alive &= ok.to(alive.device)
        contribution = factor * dt * v.to(log_w.device)
        cost_sum = cost_sum + contribution
        log_w = log_w - contribution

    accumulate(x, 0)
    for k in range(num_steps):
        x = disperse(x, s, step_cfg)
        accumulate(x, k + 1)

    log_w = log_w.masked_fill(~alive, float("-inf"))

    finite = log_w.isfinite()
    info = {
        "path_alive": alive.float().mean().item(),
        "path_cost": cost_sum[finite].mean().item() if finite.any() else float("inf"),
    }
    return x, log_w, info


def surrogate_state_cost(
    mean_and_uncertainty: Callable[[object], tuple[torch.Tensor, torch.Tensor]],
    weight: float,
    hard: bool,
    feasibility_weight: float,
    device: torch.device,
    floor: float = 1e-6,
    form: str = "surprise",
) -> Callable[[object], tuple[torch.Tensor, torch.Tensor]]:
    """The state cost of line 4 read off the surrogate: V(x) = -weight * log P(x is valid).

    With labels centred to +-0.5, the GP's posterior mean is positive where it believes a point is
    valid, so P(valid) = Phi(mu / sigma) and the cost is the surrogate's surprise, in nats, at
    finding the particle there. It is zero where the surrogate is confident and grows without bound
    where it is confident the particle is somewhere invalid.

    With `hard`, the same rule that gates endpoints, mu + lambda sigma < 0, also kills a path that
    crosses it. That is the difference the path cost buys: the feasibility constraint of line 5
    applies to the whole trajectory rather than only to where it stops, so a point reachable only by
    tunnelling through a region the surrogate is confident is invalid is no longer reachable.

    `form="surprise"` is -log P(valid) itself. It is log 2 nats per unit time wherever the surrogate
    knows nothing (mu = 0), so it charges for *unknown* space as much as for invalid space, and the
    path cost is then mostly a near-constant offset that taxes exploration -- the opposite of what
    the uncertainty acquisition asks for. `form="excess"` charges only the evidence of invalidity,
    [log P_prior(valid) - log P(valid)]_+ with P_prior = Phi(0) = 1/2: zero at the prior and where
    the surrogate leans valid, positive only where it leans invalid.
    """
    if form not in ("surprise", "excess"):
        raise ValueError(f"unknown state cost form {form!r}")

    normal = torch.distributions.Normal(0.0, 1.0)

    def cost(state) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(state)
        if n == 0:
            return torch.zeros(0, device=device), torch.ones(0, dtype=torch.bool, device=device)

        mean, uncert = mean_and_uncertainty(state)
        mean, uncert = mean.to(device), uncert.to(device).clamp_min(floor)

        p_valid = normal.cdf(mean / uncert).clamp(floor, 1.0)
        v = -weight * p_valid.log()
        if form == "excess":
            v = (v + weight * math.log(0.5)).clamp_min(0.0)

        alive = torch.ones(n, dtype=torch.bool, device=device)
        if hard:
            alive = mean + feasibility_weight * uncert >= 0

        return v, alive

    return cost


class OracleStateCost:
    """V read off the true validity of the problem instead of the surrogate: infinite (hard) or
    `weight` per unit time (soft) on invalid states. A diagnostic, not a method -- every call is a
    verifier query that no budget counts.

    Its use is as an upper bound. The surrogate cost can only prune a path once it is confident the
    wall is there, which takes queries in the wall, and in the meantime a target reached by crossing
    has usually been found. If the oracle cost cannot change where the pool goes, no surrogate cost
    will; if it can, the gap between the two is what the surrogate's lag costs.

    `interior_only` leaves the endpoints to the feasibility mask, so that the oracle prunes paths
    and does not double as a perfect endpoint filter, which would confound the comparison.
    """

    interior_only = True

    def __init__(self, valid: Callable[[object], torch.Tensor], weight: float, hard: bool, device: torch.device):
        self.valid, self.weight, self.hard, self.device = valid, weight, hard, device

    def __call__(self, state) -> tuple[torch.Tensor, torch.Tensor]:
        ok = self.valid(state).to(self.device).bool()
        v = self.weight * (~ok).float()
        return v, ok if self.hard else torch.ones_like(ok)


def tilt_weights_with_offset(
    score: torch.Tensor,
    beta: float,
    ess_min: float,
    log_offset: Optional[torch.Tensor] = None,
    feasible: Optional[torch.Tensor] = None,
    ess_over_alive: bool = False,
) -> tuple[torch.Tensor, float]:
    """Normalised weights w ∝ exp(beta * score + log_offset) on the feasible points.

    Identical to `adm.abe.tilt_weights` except that `log_offset` is held fixed while beta is
    bisected against the ESS floor. That distinction is the point: beta is an exploration
    temperature and may be lowered when the pool cannot support the tilt, whereas the path weight is
    a Radon-Nikodym derivative between two reference processes. Scaling it with beta would quietly
    change which reference the pool represents every time the ESS floor bound.

    With `ess_over_alive`, the ESS is a fraction of the paths the offset left alive rather than of
    the whole pool. A killed path is not a degenerate weight, it is a particle that does not exist,
    and counting it caps the attainable ESS at the alive fraction: with half the paths pruned an
    ESS floor of 0.5 would force beta to zero however flat the tilt. Without kills this is the
    default exactly.
    """
    if feasible is None or not feasible.any():
        feasible = torch.ones_like(score, dtype=torch.bool)

    offset = torch.zeros_like(score) if log_offset is None else log_offset
    alive = feasible & offset.isfinite()
    if not alive.any():
        # Everything was killed; fall back to the feasible set without the path weight, so that a
        # round degrades to plain ABE rather than producing an undefined distribution
        alive, offset = feasible, torch.zeros_like(score)

    def weights_at(b: float) -> torch.Tensor:
        logits = (b * score + offset).masked_fill(~alive, float("-inf"))
        return torch.softmax(logits, dim=0)

    killed = int((~offset.isfinite()).sum()) if ess_over_alive else 0
    count = max(1, len(score) - killed)

    def ess(w: torch.Tensor) -> float:
        return (1.0 / (w.pow(2).sum() * count)).item()

    weights = weights_at(beta)
    if ess(weights) >= ess_min:
        return weights, beta

    lo, hi = 0.0, beta
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if ess(weights_at(mid)) >= ess_min:
            lo = mid
        else:
            hi = mid

    return weights_at(lo), lo
