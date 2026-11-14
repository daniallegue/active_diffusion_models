from dataclasses import dataclass
import math
from typing import Callable, Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


Clock = Callable[[int, torch.device], torch.Tensor]


def positive_stable(a: float, n: int, device: torch.device) -> torch.Tensor:
    """Samples of the positive a-stable law with Laplace transform exp(-s^a), for a in (0, 1], with
    Kanter's representation."""
    if a >= 1:
        return torch.ones(n, device=device)

    u = math.pi * torch.rand(n, device=device).clamp(1e-6, 1 - 1e-6)
    e = torch.empty(n, device=device).exponential_()
    return (torch.sin(a * u) / torch.sin(u) ** (1 / a)) * (torch.sin((1 - a) * u) / e) ** ((1 - a) / a)


CLOCKS = ("gaussian", "levy", "student", "poisson")


def make_clock(
    kind: str,
    dt: float,
    alpha: float = 1.5,
    df: float = 2.0,
    jump_rate: float = 0.2,
    jump_clock: float = 1.0,
) -> Clock:
    """The per-step clock of the CTMC reference: how much reference time each particle runs for
    in a step of nominal length dt (in units of `disperse_time`), shared by all its coordinates.

    The continuous references of `adm.abe.sample_reference` are Brownian motion with a random
    variance factor, i.e., Brownian motion on a random clock, and these are the same clocks:

        gaussian: dt, the plain CTMC.
        levy:     dt^{2/alpha} A, A positive (alpha / 2)-stable, so that n steps sum to tau^{2/alpha} A
                  in law and splitting is exact. A coordinate then stays put over tau with
                  probability exp(-tau lambda^{alpha/2}), so lambda^{alpha/2} is its effective rate.
        student:  dt df / chi2_df per step. Not stable, so a split dispersal has a different endpoint
                  law than a single step, exactly as for the continuous Student-t.
        poisson:  dt plus `jump_clock` for each of N ~ Poisson(jump_rate dt) jumps, a compound
                  Poisson subordinator, so splitting is exact.

    Because the clock is shared, a long draw changes many coordinates of a particle at once, which
    no per-coordinate rate does: that is what the heavy tails buy on a categorical state.
    """
    if kind == "gaussian":
        return lambda n, device: torch.full((n,), dt, device=device)

    if kind == "levy":
        if not (0 < alpha <= 2):
            raise ValueError(f"alpha must be in (0, 2], got {alpha}")
        return lambda n, device: dt ** (2 / alpha) * positive_stable(alpha / 2, n, device)

    if kind == "student":
        chi2 = torch.distributions.Chi2(torch.tensor(float(df)))
        return lambda n, device: dt * df / chi2.sample((n,)).to(device)

    if kind == "poisson":
        return lambda n, device: dt + jump_clock * torch.poisson(torch.full((n,), jump_rate * dt, device=device))

    raise ValueError(f"unknown clock {kind!r}, expected one of {CLOCKS}")


@dataclass
class CategoricalState:
    """A batch of factorised categorical states, flattened over coordinates.

    `tokens[m]` is the class of coordinate m, `field[m]` its field (e.g., atom type, charge or bond),
    and `owner[m]` the particle it belongs to. `links` holds directed coordinate pairs (both
    directions) along which the twist passes messages, e.g., a bond to its two atoms. Particles may
    have different numbers of coordinates, as molecules with different numbers of atoms do.
    """

    tokens: torch.Tensor  # (M,) long
    field: torch.Tensor   # (M,) long
    owner: torch.Tensor   # (M,) long
    links: torch.Tensor   # (2, L) long
    num_particles: int

    def __len__(self) -> int:
        return self.num_particles

    def with_tokens(self, tokens: torch.Tensor) -> "CategoricalState":
        return CategoricalState(tokens, self.field, self.owner, self.links, self.num_particles)

    def subset(self, idx: torch.Tensor) -> tuple["CategoricalState", torch.Tensor]:
        """The states of particles `idx`, renumbered 0..len(idx)-1 in that order, and the
        coordinates they consist of, to select the matching tokens of other steps with."""
        idx = idx.to(self.owner.device)
        new_id = torch.full((self.num_particles,), -1, dtype=torch.long, device=self.owner.device)
        new_id[idx] = torch.arange(len(idx), device=idx.device)

        coords = (new_id[self.owner] >= 0).nonzero().squeeze(1)
        coords = coords[torch.argsort(new_id[self.owner[coords]], stable=True)]
        new_coord = torch.full_like(self.owner, -1)
        new_coord[coords] = torch.arange(len(coords), device=coords.device)

        keep = (new_coord[self.links[0]] >= 0) & (new_coord[self.links[1]] >= 0)
        links = new_coord[self.links[:, keep]]
        state = CategoricalState(
            self.tokens[coords], self.field[coords], new_id[self.owner[coords]], links, len(idx)
        )
        return state, coords


@dataclass
class ClockMap:
    """A linear, entrywise non-negative map from independent clock draws xi at `num_sites` sites to
    the clock of every coordinate:

        T_site  =  K xi                                  (K block diagonal over particles)
        T_m     =  sum_c weight[m, c] T_site[site[m, c]]  (two sites per coordinate)

    Each site draws xi from the clock law of `make_clock`, so for a K that is fixed along the path,
    n steps sum to K (sum_k xi_k), and a Levy or Poisson clock splits exactly. `shared_clock_map` is
    one site per particle with K = I, the clock all coordinates of a particle share;
    `molecular_clock_map` smooths per-atom and per-bond draws with heat kernels of the molecule.
    """

    site_owner: torch.Tensor  # (S,) particle of each site
    rows: torch.Tensor        # (E,) entries of K
    cols: torch.Tensor        # (E,)
    vals: torch.Tensor        # (E,)
    site: torch.Tensor        # (M, 2) sites each coordinate averages
    weight: torch.Tensor      # (M, 2)

    @property
    def num_sites(self) -> int:
        return len(self.site_owner)

    def apply(self, xi: torch.Tensor) -> torch.Tensor:
        """Coordinate clocks of shape (M,) from site draws `xi` of shape (S,)."""
        t_site = torch.zeros_like(xi).index_add(0, self.rows, self.vals * xi[self.cols])
        return (self.weight * t_site[self.site]).sum(1)


def shared_clock_map(state: CategoricalState) -> ClockMap:
    """One clock per particle, shared by all of its coordinates."""
    n, device = len(state), state.owner.device
    sites = torch.arange(n, device=device)
    return ClockMap(
        sites, sites, sites, torch.ones(n, device=device),
        torch.stack([state.owner, state.owner], dim=1),
        torch.full((len(state.owner), 2), 0.5, device=device),
    )


def heat_kernel(
    site_owner: torch.Tensor, edges: torch.Tensor, s: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Entries (rows, cols, vals) of K = exp(-s L), L = D - A the Laplacian of the undirected
    graph on the sites with `edges` of shape (2, E) (each pair once or twice, never across particles).

    -L has non-negative off-diagonal entries, so K is entrywise non-negative, and L is symmetric with
    zero row sums, so K is doubly stochastic: every clock is a weighted average of the draws around
    it, and the clock mass of a particle is conserved. Particles are batched by size and their
    kernels computed densely, which molecules are small enough for.
    """
    device = site_owner.device
    num_sites = len(site_owner)
    if num_sites == 0:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty, torch.zeros(0, device=device)

    order = torch.argsort(site_owner, stable=True)
    counts = torch.bincount(site_owner)
    starts = torch.cumsum(counts, 0) - counts
    local = torch.empty_like(site_owner)
    local[order] = torch.arange(num_sites, device=device) - starts[site_owner[order]]

    rows, cols, vals = [], [], []
    for n in counts.unique().tolist():
        if n == 0:
            continue
        particles = (counts == n).nonzero().squeeze(1)
        batch = torch.full((len(counts),), -1, dtype=torch.long, device=device)
        batch[particles] = torch.arange(len(particles), device=device)

        adj = torch.zeros(len(particles), n, n, device=device)
        if edges.shape[1] > 0:
            i, j = edges
            sel = batch[site_owner[i]] >= 0
            i, j = i[sel], j[sel]
            b = batch[site_owner[i]]
            adj[b, local[i], local[j]] = 1.0
            adj[b, local[j], local[i]] = 1.0
        adj.diagonal(dim1=1, dim2=2).zero_()
        lap = torch.diag_embed(adj.sum(2)) - adj
        kern = torch.linalg.matrix_exp(-s * lap).clamp_min(0)

        # Global site ids of each particle's sites, in local order
        members = order[(starts[particles][:, None] + torch.arange(n, device=device)[None, :])]
        rows.append(members[:, :, None].expand(-1, n, n).reshape(-1))
        cols.append(members[:, None, :].expand(-1, n, n).reshape(-1))
        vals.append(kern.reshape(-1))

    return torch.cat(rows), torch.cat(cols), torch.cat(vals)


def molecular_clock_map(
    state: CategoricalState,
    s: float,
    no_bond: int,
    atom_field: int = 0,
    bond_field: int = 2,
    charge_field: Optional[int] = 1,
) -> ClockMap:
    """Clocks smoothed over the molecular graph of `state` with heat time s.

    Sites are the atoms and the present bonds (token != `no_bond`), each drawing its own xi.

        atoms:         T = exp(-s L0) xi_atoms, L0 the graph Laplacian of the present bonds.
        present bonds: T = exp(-s L1) xi_bonds, L1 the Laplacian of the line graph, i.e., of bonds
                       that share an atom. This is the down Hodge Laplacian B1^T B1 with the
                       orientation signs dropped: those suit flows, but make the kernel of a scalar
                       clock negative.
        charges:       their atom's clock.
        absent pairs:  the mean of their two atoms' clocks, so that a bond appears where its atoms
                       are active.

    A large draw at an atom spreads over its neighbourhood with heat-kernel decay, so long clocks
    rewrite a region of the molecule rather than coordinates scattered over all of it. At s = 0
    every site keeps its own draw. The map depends on the bonds of `state`; frozen at the start of
    a dispersal it keeps the splitting exact.
    """
    device = state.owner.device
    m = len(state.tokens)
    atoms = (state.field == atom_field).nonzero().squeeze(1)
    bonds = (state.field == bond_field).nonzero().squeeze(1)

    # Endpoints of each bond coordinate, from its links to atoms
    is_atom = torch.zeros(m, dtype=torch.bool, device=device)
    is_atom[atoms] = True
    src, dst = state.links
    to_atom = (state.field[src] == bond_field) & is_atom[dst]
    b_src, b_atom = src[to_atom], dst[to_atom]
    b_atom = b_atom[torch.argsort(b_src, stable=True)]
    b_src = b_src.sort(stable=True).values
    if len(b_src) != 2 * len(bonds) or not torch.equal(b_src[0::2], b_src[1::2]):
        raise ValueError("every bond coordinate must be linked to exactly two atoms")
    ends = b_atom.view(-1, 2)                                 # (B, 2), ordered like `bonds`

    # Sites: atoms first, then present bonds
    atom_site = torch.full((m,), -1, dtype=torch.long, device=device)
    atom_site[atoms] = torch.arange(len(atoms), device=device)
    present = state.tokens[bonds] != no_bond
    present_bonds = bonds[present]
    bond_site = torch.full((m,), -1, dtype=torch.long, device=device)
    bond_site[present_bonds] = len(atoms) + torch.arange(len(present_bonds), device=device)
    site_owner = torch.cat([state.owner[atoms], state.owner[present_bonds]])

    # L0 on atoms, along present bonds
    present_ends = atom_site[ends[present]]                   # (P, 2)
    r0, c0, v0 = heat_kernel(state.owner[atoms], present_ends.T, s)

    # L1 on present bonds, along shared atoms: pairs of incidences with the same atom. Sorted by
    # atom, the incidences of an atom are contiguous, so comparing at every shift up to the largest
    # degree finds all pairs without a quadratic matrix over the batch
    inc_bond = torch.arange(len(present_bonds), device=device).repeat_interleave(2)
    inc_atom = present_ends.reshape(-1)
    by_atom = torch.argsort(inc_atom, stable=True)
    inc_atom, inc_bond = inc_atom[by_atom], inc_bond[by_atom]
    max_degree = int(torch.bincount(inc_atom).max()) if len(inc_atom) > 0 else 0
    pairs = [torch.zeros(2, 0, dtype=torch.long, device=device)]
    for d in range(1, max_degree):
        k = (inc_atom[:-d] == inc_atom[d:]).nonzero().squeeze(1)
        pairs.append(torch.stack([inc_bond[k], inc_bond[k + d]]))
    line_edges = torch.cat(pairs, dim=1)
    r1, c1, v1 = heat_kernel(state.owner[present_bonds], line_edges, s)
    offset = len(atoms)

    # Coordinates average two sites: themselves twice, or the two atoms of an absent pair
    site = torch.zeros(m, 2, dtype=torch.long, device=device)
    site[atoms] = atom_site[atoms][:, None]
    if charge_field is not None:
        charges = (state.field == charge_field).nonzero().squeeze(1)
        to_charge_atom = (state.field[src] == charge_field) & is_atom[dst]
        charge_atom = torch.full((m,), -1, dtype=torch.long, device=device)
        charge_atom[src[to_charge_atom]] = dst[to_charge_atom]
        if (charge_atom[charges] < 0).any():
            raise ValueError("every charge coordinate must be linked to its atom")
        site[charges] = atom_site[charge_atom[charges]][:, None]
    site[present_bonds] = bond_site[present_bonds][:, None]
    site[bonds[~present]] = atom_site[ends[~present]]

    return ClockMap(
        site_owner,
        torch.cat([r0, r1 + offset]), torch.cat([c0, c1 + offset]), torch.cat([v0, v1]),
        site, torch.full((m, 2), 0.5, device=device),
    )


class CTMCReference:
    """The frozen reference: per field f a jump rate lambda_f and a jump law pi_f over its classes.

    Fields have different numbers of classes, so laws are padded to the largest with zeros, i.e.,
    -inf in log space, and a padded class is never reachable.
    """

    def __init__(self, rates: list[float], pis: list[torch.Tensor], device: Optional[torch.device] = None):
        if len(rates) != len(pis):
            raise ValueError(f"{len(rates)} rates for {len(pis)} fields")

        self.num_classes = [len(p) for p in pis]
        k_max = max(self.num_classes)
        pi = torch.zeros(len(pis), k_max)
        for f, p in enumerate(pis):
            p = torch.as_tensor(p, dtype=torch.float32)
            if (p < 0).any() or p.sum() <= 0:
                raise ValueError(f"the jump law of field {f} is not a distribution")
            pi[f, : len(p)] = p / p.sum()

        self.rates = torch.as_tensor(rates, dtype=torch.float32, device=device)
        if (self.rates < 0).any():
            raise ValueError(f"rates must be non-negative, got {rates}")

        self.pi = pi.to(device)
        self.log_pi = self.pi.log()

    @property
    def k_max(self) -> int:
        return self.pi.shape[1]

    @staticmethod
    def marginals(state: CategoricalState, num_classes: list[int], smoothing: float = 1.0) -> list[torch.Tensor]:
        """Per-field class frequencies of `state` with `smoothing` pseudo-counts, e.g., to freeze
        pi to the pretrained model's marginals once, so that a jump rarely creates a bond out of
        nothing, as a uniform law over five bond types would most of the time."""
        pis = []
        for f, k in enumerate(num_classes):
            tok = state.tokens[state.field == f]
            pis.append(torch.bincount(tok.cpu(), minlength=k).float() + smoothing)

        return pis

    def to(self, device: torch.device) -> "CTMCReference":
        self.rates, self.pi, self.log_pi = self.rates.to(device), self.pi.to(device), self.log_pi.to(device)
        return self

    def log_transition(self, state: CategoricalState, dt: float | torch.Tensor) -> torch.Tensor:
        """log p_dt(k | x_m) for every coordinate m and class k, of shape (M, k_max). `dt` is one
        time for all coordinates, or one per coordinate of shape (M,), e.g., from a `ClockMap`."""
        if torch.is_tensor(dt):
            dt = dt.to(state.owner.device)
        stay = torch.exp(-self.rates[state.field] * dt)          # (M,)
        p = (1 - stay)[:, None] * self.pi[state.field]           # (M, K)
        p[torch.arange(len(state.tokens), device=p.device), state.tokens] += stay
        return p.log()

    def endpoint(self, state: CategoricalState, tau: float) -> torch.Tensor:
        """The exact law p_tau(k | x_m) of every coordinate after time tau, of shape (M, k_max)."""
        return self.log_transition(state, tau).exp()


def twisted_step(
    reference: CTMCReference,
    state: CategoricalState,
    dt: float | torch.Tensor,
    logits: Optional[torch.Tensor] = None,
) -> tuple[CategoricalState, torch.Tensor]:
    """One step of the reference over time `dt` (one per coordinate, or shared), twisted by `logits`
    of shape (M, k_max) if given, and the per-particle log correction log p_dt(x' | x) - log q(x' | x)
    of shape (num_particles,).

    With a random clock, the correction is conditional on the clock's draw. The clock itself is
    drawn from its reference law and never twisted, so the weight is the importance weight of the
    augmented path (x, clock), and with it exact for x alone."""
    log_p = reference.log_transition(state, dt)
    if logits is None:
        new = torch.distributions.Categorical(logits=log_p).sample()
        return state.with_tokens(new), torch.zeros(len(state), device=log_p.device)

    log_q = log_p + logits
    new = torch.distributions.Categorical(logits=log_q).sample()
    corr = step_correction(log_p, logits, new)
    return state.with_tokens(new), per_particle(corr, state.owner, len(state))


def step_correction(log_p: torch.Tensor, logits: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
    """Per-coordinate log p(x'_m) - log q(x'_m) = logsumexp_k (log p + l) - l(x'_m).

    Padded classes have log p = -inf and drop out of the logsumexp whatever their logit."""
    log_norm = torch.logsumexp(log_p + logits, dim=1)
    return log_norm - logits.gather(1, new[:, None]).squeeze(1)


def per_particle(values: torch.Tensor, owner: torch.Tensor, n: int) -> torch.Tensor:
    return torch.zeros(n, device=values.device, dtype=values.dtype).index_add(0, owner, values)


class TwistNet(nn.Module):
    """Logits l_m(k; x, t) of the twist, from message passing over the coordinates of each state.

    Every coordinate embeds its (field, class), the time, and its own clock relative to the nominal
    step, then exchanges messages with its linked coordinates and with the mean of its particle for
    `layers` rounds. The clock lets the twist act differently where a step is long, which changes
    many coordinates at once, than where it is short. The output layer starts at zero, so an
    untrained twist is the reference exactly.
    """

    def __init__(self, num_classes: list[int], hidden: int = 64, layers: int = 3):
        super().__init__()
        self.num_fields = len(num_classes)
        self.k_max = max(num_classes)
        self.embed = nn.Embedding(self.num_fields * self.k_max, hidden)
        self.time = nn.Linear(3, hidden)
        self.layers = nn.ModuleList(
            nn.Sequential(nn.Linear(3 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.head = nn.Linear(hidden, self.num_fields * self.k_max)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, state: CategoricalState, t: float, log_clock: Optional[torch.Tensor] = None) -> torch.Tensor:
        """`log_clock` is log(clock / dt) per coordinate, zero (the default) for the plain CTMC."""
        m = len(state.tokens)
        device = state.tokens.device
        tt = torch.tensor([math.sin(math.pi * t), math.cos(math.pi * t)], device=device).expand(m, 2)
        if log_clock is None:
            lc = torch.zeros(m, 1, device=device)
        else:
            lc = log_clock.to(device).clamp(-10, 10)[:, None]
        h = self.embed(state.field * self.k_max + state.tokens) + self.time(torch.cat([tt, lc], dim=1))

        src, dst = state.links
        for layer, norm in zip(self.layers, self.norms):
            msg = torch.zeros_like(h).index_add(0, dst, h[src])
            deg = torch.zeros(m, device=device).index_add(0, dst, torch.ones_like(dst, dtype=h.dtype))
            msg = msg / deg.clamp_min(1)[:, None]

            pooled = per_particle_mean(h, state.owner, len(state))[state.owner]
            h = norm(h + layer(torch.cat([h, msg, pooled], dim=1)))

        out = self.head(h).view(m, self.num_fields, self.k_max)
        return out[torch.arange(m, device=device), state.field]


def per_particle_mean(h: torch.Tensor, owner: torch.Tensor, n: int) -> torch.Tensor:
    total = torch.zeros(n, h.shape[1], device=h.device, dtype=h.dtype).index_add(0, owner, h)
    count = torch.zeros(n, device=h.device, dtype=h.dtype).index_add(0, owner, torch.ones_like(owner, dtype=h.dtype))
    return total / count.clamp_min(1)[:, None]


@dataclass
class CTMCRecord:
    """The tokens of a simulated path at every step, on the coordinate layout of `state`, and the
    clock each coordinate ran for in each step."""

    state: CategoricalState
    tokens: list[torch.Tensor]  # num_steps + 1 tensors of shape (M,)
    clocks: list[torch.Tensor]  # num_steps tensors of shape (M,)
    dt: float                   # nominal step

    @property
    def num_steps(self) -> int:
        return len(self.tokens) - 1


class CTMCDisperser:
    """A `path_cost.Disperser` that takes one (twisted) step of the reference per call.

    `simulate_path` charges V and kills particles exactly as for the continuous references; this
    keeps the twist's correction on the side, in `log_ratio`, for the caller to add to `log_w`, and
    records the tokens and clocks of every step for training the twist.

    `encode` and `decode` translate between the latents and their categorical state, such that
    everything continuous, e.g., positions, is carried through unchanged. `clock` is a `make_clock`
    of nominal step tau / num_steps; the plain CTMC without one.

    `clock_map(state)` builds the `ClockMap` that spreads the clock's site draws over coordinates,
    one shared clock per particle without one. With `freeze_clock_map` it is built once from the
    start of the path, so that it is fixed and splitting stays exact; otherwise it follows the state
    at every step, e.g., the bonds as they change. Either way the clock is drawn from its reference
    law given the state, never twisted, so the weights stay exact.
    """

    def __init__(self, reference: CTMCReference, encode, decode, num_steps: int, tau: float,
                 twist: Optional[TwistNet] = None, clock: Optional[Clock] = None, clock_kind: str = "gaussian",
                 clock_map: Optional[Callable[[CategoricalState], ClockMap]] = None, freeze_clock_map: bool = True):
        self.reference, self.encode, self.decode = reference, encode, decode
        self.num_steps, self.dt, self.twist = num_steps, tau / num_steps, twist
        self.clock = clock if clock is not None else make_clock("gaussian", self.dt)
        self.clock_map_fn = clock_map if clock_map is not None else shared_clock_map
        self.freeze_clock_map = freeze_clock_map
        self.clock_map: Optional[ClockMap] = None
        self.k = 0
        self.log_ratio: Optional[torch.Tensor] = None
        self.record: Optional[CTMCRecord] = None

        if clock_kind == "student" and num_steps > 1:
            warnings.warn(
                f"the Student-t clock is not stable, so splitting the CTMC dispersal into {num_steps} "
                "steps changes its endpoint law, as for the continuous Student-t reference",
                RuntimeWarning,
                stacklevel=2,
            )

    @torch.no_grad()
    def __call__(self, x, scale: float, cfg):
        state = self.encode(x)
        device = state.tokens.device
        if self.record is None:
            self.record = CTMCRecord(state, [state.tokens], [], self.dt)
            self.log_ratio = torch.zeros(len(state), device=device)

        if self.clock_map is None or not self.freeze_clock_map:
            self.clock_map = self.clock_map_fn(state)
        clock = self.clock_map.apply(self.clock(self.clock_map.num_sites, device))
        logits = None
        if self.twist is not None:
            logits = self.twist(state, self.k / self.num_steps, (clock / self.dt).log())
        new, corr = twisted_step(self.reference, state, clock, logits)

        self.k += 1
        self.log_ratio = self.log_ratio + corr
        self.record.tokens.append(new.tokens)
        self.record.clocks.append(clock)
        return self.decode(x, new.tokens)


def path_log_ratio(
    twist: TwistNet,
    reference: CTMCReference,
    record: CTMCRecord,
    idx: torch.Tensor,
) -> torch.Tensor:
    """sum_k log p_dt / q_theta along the recorded paths of particles `idx`, differentiable in theta."""
    state, coords = record.state.subset(idx)
    total = torch.zeros(len(idx), device=state.tokens.device)
    for k in range(record.num_steps):
        s = state.with_tokens(record.tokens[k][coords])
        clock = record.clocks[k][coords.to(record.clocks[k].device)]
        logits = twist(s, k / record.num_steps, (clock / record.dt).log())
        log_p = reference.log_transition(s, clock)
        corr = step_correction(log_p, logits, record.tokens[k + 1][coords])
        total = total + per_particle(corr, s.owner, len(idx))

    return total


def fit_twist(
    twist: TwistNet,
    optimizer: torch.optim.Optimizer,
    reference: CTMCReference,
    record: CTMCRecord,
    base_log_w: torch.Tensor,
    steps: int,
    batch_size: int,
) -> float:
    """Fits the twist on recorded paths with the log-variance loss against the round's target.

    `base_log_w` is the part of the target's log density that does not depend on the twist, i.e.,
    -int V + beta alpha(X_1) relative to the reference, with -inf on paths the target does not
    contain (killed or infeasible). Returns the mean loss."""
    alive = base_log_w.isfinite().nonzero().squeeze(1)
    if len(alive) < 2 or steps <= 0:
        return float("nan")

    base = base_log_w.detach()
    twist.train()
    loss_sum = 0.0
    for _ in range(steps):
        idx = alive[torch.randperm(len(alive), device=alive.device)[:batch_size]]
        log_ratio = base[idx] + path_log_ratio(twist, reference, record, idx)
        loss = log_ratio.var()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(twist.parameters(), 1.0)
        optimizer.step()
        loss_sum += loss.item()

    twist.eval()
    return loss_sum / steps
