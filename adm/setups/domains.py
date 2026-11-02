"""Reachability domains: the geometry half of the corridor family, as data rather than branches.

A `Domain` is a valid region, a route through it, and the derived quantities the path-cost
hypothesis is stated in. The hypothesis is that a state cost on ABE's reference buys something no
terminal reward can, and it only has room to do so when **the geodesic distance through the valid
set and the ambient distance disagree**. That disagreement is `detour_ratio`, and making it a dial
rather than a hand-built shape is the point of this module: the claim becomes a curve over the
ratio instead of a single anecdote.

Everything here is pure geometry on tensors, so it imports nothing from diffusiongym and can be
exercised directly.

The families, all sharing the same S_0 so one pre-trained checkpoint serves every one of them:

    corridor(w, l)      S_0 and A joined along the x-axis. detour_ratio == 1: the shortcut and the
                        valid route are the same line, so no state cost can discriminate. This is
                        the control, and the reason the original benchmark cannot test the claim.
    islands(gap)        w = 0. A is unreachable by any valid path; a hard cost forbids everything.
    hook(w, l, gap)     A sits above S_0, Euclidean-near but reachable only the long way round.
    switchback(k, ...)  A serpentine with k folds. detour_ratio grows with k, which is the axis the
                        hypothesis should be swept along.
"""

from typing import Callable
from dataclasses import dataclass
import math

import torch


Box = tuple[float, float, float, float]      # xmin, xmax, ymin, ymax
Point = tuple[float, float]


def in_box(xy: torch.Tensor, box: Box) -> torch.Tensor:
    xmin, xmax, ymin, ymax = box
    x, y = xy[:, 0], xy[:, 1]
    return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)


def union_box(a: Box, b: Box) -> Box:
    return (min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3]))


@dataclass(frozen=True)
class Domain:
    """A valid region as a union of boxes, with a route through it from S_0 to A.

    `boxes` must contain `source` and `target`; the ones between them are the route's corridors.
    `route` is the centre line, used to measure progress as arclength, which is the only honest
    measure once a domain doubles back on itself.
    """

    name: str
    source: Box
    target: Box
    boxes: tuple[Box, ...]
    route: tuple[Point, ...]

    def __post_init__(self):
        if len(self.route) < 2:
            raise ValueError("a route needs at least two points")

        if len(self.boxes) == 0:
            raise ValueError("a domain needs at least one valid box")

    # -- membership ---------------------------------------------------------------------------

    def contains(self, xy: torch.Tensor) -> torch.Tensor:
        """Whether each point lies in the valid region."""
        inside = torch.zeros(len(xy), dtype=torch.bool, device=xy.device)
        for box in self.boxes:
            inside = inside | in_box(xy, box)

        return inside

    def in_target(self, xy: torch.Tensor) -> torch.Tensor:
        return in_box(xy, self.target)

    @property
    def span(self) -> Box:
        """The bounding box of S_0 and A together."""
        return union_box(self.source, self.target)

    def crossed(self, xy: torch.Tensor) -> torch.Tensor:
        """Points in the forbidden space between S_0 and A.

        Mass here can only have crossed the valid set rather than travelled it, which is the
        signature a path cost is meant to remove and a terminal reward cannot see. Derived from the
        geometry rather than declared, so it means the same thing for every family.
        """
        return in_box(xy, self.span) & ~self.contains(xy)

    # -- distances ----------------------------------------------------------------------------

    @property
    def geodesic_length(self) -> float:
        return sum(math.dist(a, b) for a, b in zip(self.route[:-1], self.route[1:]))

    @property
    def euclidean_distance(self) -> float:
        return math.dist(self.route[0], self.route[-1])

    @property
    def detour_ratio(self) -> float:
        """How much longer the valid route is than the straight line. 1 means they coincide, and a
        state cost then has nothing to discriminate."""
        return self.geodesic_length / max(self.euclidean_distance, 1e-9)

    def progress(self, xy: torch.Tensor) -> torch.Tensor:
        """Normalised arclength of the projection onto the route, clipped to [0, 1].

        Measured from the mouth, i.e. from the end of the first segment, which lies inside S_0. The
        x coordinate would do for a straight corridor but scores a point just outside S_0 the same
        as one that has come all the way round a hook.
        """
        best = torch.full((len(xy),), float("inf"), device=xy.device)
        arc = torch.zeros(len(xy), device=xy.device)
        travelled = 0.0

        for a, b in zip(self.route[:-1], self.route[1:]):
            pa = torch.tensor(a, device=xy.device, dtype=xy.dtype)
            pb = torch.tensor(b, device=xy.device, dtype=xy.dtype)
            seg = pb - pa
            length = float(seg.norm().clamp_min(1e-9))
            t = (((xy - pa) @ seg) / (length ** 2)).clamp(0, 1)
            d = (xy - (pa + t[:, None] * seg)).norm(dim=1)
            closer = d < best
            best = torch.where(closer, d, best)
            arc = torch.where(closer, travelled + t * length, arc)
            travelled += length

        mouth = math.dist(self.route[0], self.route[1])
        return ((arc - mouth) / max(self.geodesic_length - mouth, 1e-9)).clamp(0, 1)

    # -- obstacles ----------------------------------------------------------------------------

    def wall_thickness(self, samples: int = 4001) -> float:
        """Length of the longest forbidden stretch on the straight line from S_0 to A.

        This is the `b` in the detection rate min(1, (n-1) b/d) of a dispersal split into n
        increments, so it is what should set `--abe_disperse_steps`: the reference cannot charge for
        a wall it never lands in. Zero when the straight line is valid throughout, as in the
        corridor, where no number of increments changes anything.
        """
        a = torch.tensor(self._centre(self.source))
        b = torch.tensor(self._centre(self.target))
        ts = torch.linspace(0.0, 1.0, samples)
        line = a + ts[:, None] * (b - a)
        blocked = ~self.contains(line)

        longest = run = 0
        for flag in blocked.tolist():
            run = run + 1 if flag else 0
            longest = max(longest, run)

        return longest / (samples - 1) * float((b - a).norm())

    def required_steps(self, detect: float = 0.9) -> int:
        """Increments needed to notice the wall with probability `detect`, from the same relation.

        A straight-line answer to "what should --abe_disperse_steps be for this domain". Returns 1
        when there is no wall, since no resolution helps.
        """
        b = self.wall_thickness()
        if b <= 0:
            return 1

        d = math.dist(self._centre(self.source), self._centre(self.target))
        return max(2, math.ceil(1 + detect * d / b))

    @staticmethod
    def _centre(box: Box) -> Point:
        xmin, xmax, ymin, ymax = box
        return (0.5 * (xmin + xmax), 0.5 * (ymin + ymax))

    @property
    def view_extent(self) -> Box:
        xs = [v for box in self.boxes for v in box[:2]]
        ys = [v for box in self.boxes for v in box[2:]]
        return (min(xs) - 0.5, max(xs) + 0.5, min(ys) - 0.5, max(ys) + 0.5)

    def summary(self) -> dict[str, float]:
        return {
            "detour_ratio": self.detour_ratio,
            "geodesic_length": self.geodesic_length,
            "euclidean_distance": self.euclidean_distance,
            "wall_thickness": self.wall_thickness(),
            "required_steps": float(self.required_steps()),
        }


# -- families ------------------------------------------------------------------------------------

def _source_box(x0: float, s: float) -> Box:
    return (x0 - s, x0, -s / 2, s / 2)


def corridor(width: float, length: float, island: float = 1.0, x0: float = -1.5) -> Domain:
    """The original geometry: S_0 and A on the x-axis, joined by a channel of width `width`.

    `width = 0` removes the channel and gives `islands`. detour_ratio is 1 by construction, which is
    why this family is the control rather than a test.
    """
    s = island
    src = _source_box(x0, s)
    x1 = x0 + length
    tgt = (x1, x1 + s, -s / 2, s / 2)
    boxes = [src, tgt]
    if width > 0:
        boxes.insert(1, (x0, x1, -width / 2, width / 2))

    route = ((x0 - s / 2, 0.0), (x0, 0.0), (x1, 0.0), (x1 + s / 2, 0.0))
    name = "corridor" if width > 0 else "islands"
    return Domain(name=name, source=src, target=tgt, boxes=tuple(boxes), route=route)


def islands(length: float, island: float = 1.0, x0: float = -1.5) -> Domain:
    """Two disconnected islands separated by `length`. No valid route exists, so a hard state cost
    forbids every path: the honest outcome is that nothing is ever reached."""
    return corridor(0.0, length, island, x0)


def hook(width: float, length: float, gap: float, island: float = 1.0, x0: float = -1.5) -> Domain:
    """A above S_0, Euclidean-near but reachable only by going right, up, and back left.

    The smallest domain in which a state cost and a terminal reward disagree while both still end in
    A, which is the case the hypothesis is about.
    """
    s, w = island, width
    if w <= 0:
        raise ValueError("the hook needs a positive width: its point is that a valid detour exists")

    if gap <= 0:
        raise ValueError(f"gap must be positive, got {gap}")

    src = _source_box(x0, s)
    y0 = s / 2 + gap
    tgt = (x0 - s, x0, y0, y0 + s)
    h = y0 + s / 2
    boxes = (
        src,
        (x0, x0 + length, -w / 2, w / 2),                  # arm out
        (x0 + length - w, x0 + length, -w / 2, h + w / 2),  # riser
        (x0 - s, x0 + length, h - w / 2, h + w / 2),        # arm back
        tgt,
    )
    xr = x0 + length - w / 2
    route = ((x0 - s / 2, 0.0), (xr, 0.0), (xr, h), (x0 - s / 2, h))
    return Domain(name="hook", source=src, target=tgt, boxes=boxes, route=route)


def switchback(
    folds: int,
    width: float,
    length: float,
    gap: float,
    island: float = 1.0,
    x0: float = -1.5,
) -> Domain:
    """A serpentine with `folds` rungs stacked above S_0, with A on top.

    The parametric family: each fold adds about `length` to the valid route while adding only `gap`
    to the straight line, so the fold count dials `detour_ratio` up. That turns the hypothesis into
    a curve rather than an anecdote -- the benefit of a state cost should grow with the ratio and
    vanish at ratio 1, where the corridor sits.

    The dial is monotone at fixed parity, not across it; see the comment in the body. Sweep odd
    fold counts, or order a sweep by the measured ratio.
    """
    if folds < 1:
        raise ValueError(f"folds must be at least 1, got {folds}")

    if width <= 0 or gap <= 0:
        raise ValueError("switchback needs positive width and gap")

    s, w = island, width
    src = _source_box(x0, s)
    left, right = x0, x0 + length

    boxes: list[Box] = [src, (x0, right + w / 2, -w / 2, w / 2)]       # S_0 and the arm out
    route: list[Point] = [(x0 - s / 2, 0.0)]

    # Sides alternate, with the last riser on the right so the walk always has a final rung to
    # traverse before climbing to A above the left end.
    #
    # `detour_ratio` is monotone in `folds` only at fixed parity: a traversal is worth `length`
    # while a fold adds only `gap` of height, so changing parity moves the ratio by far more than
    # one fold does. Sweep odd fold counts, or better, sweep the measured ratio itself -- it is the
    # quantity the hypothesis is stated in, and the fold count is only a way of producing it.
    prev_y = 0.0
    for k in range(folds):
        y = s / 2 + gap * (k + 1)
        riser_x = right if (folds - 1 - k) % 2 == 0 else left
        boxes.append((riser_x - w / 2, riser_x + w / 2, prev_y - w / 2, y + w / 2))   # riser
        boxes.append((left - w / 2, right + w / 2, y - w / 2, y + w / 2))             # rung
        route.append((riser_x, prev_y))
        route.append((riser_x, y))
        prev_y = y

    # Always finish above S_0, so that A stays Euclidean-near the source whatever the fold count.
    # When the last riser was climbed on the left the walk is already there; otherwise it continues
    # along the rung it is standing on, which is valid for its whole span.
    end_x = left
    top = prev_y + gap
    boxes.append((end_x - w / 2, end_x + w / 2, prev_y - w / 2, top + s / 2))         # final riser
    tgt = (end_x - s / 2, end_x + s / 2, top, top + s)
    boxes.append(tgt)
    if route[-1] != (end_x, prev_y):
        route.append((end_x, prev_y))

    route.append((end_x, top + s / 2))

    return Domain(name=f"switchback{folds}", source=src, target=tgt,
                  boxes=tuple(boxes), route=tuple(route))


FAMILIES = {
    "corridor": corridor,
    "islands": islands,
    "hook": hook,
    "switchback": switchback,
}


def build(name: str, **kwargs) -> Domain:
    """Builds a domain by family name, ignoring arguments the family does not take, so one sweep
    config can drive every family."""
    if name not in FAMILIES:
        raise ValueError(f"unknown domain family {name!r}; have {sorted(FAMILIES)}")

    import inspect

    fn = FAMILIES[name]
    allowed = set(inspect.signature(fn).parameters)
    return fn(**{k: v for k, v in kwargs.items() if k in allowed})


# -- the model's space is not the problem's space ------------------------------------------------

@dataclass(frozen=True)
class Decoder:
    """phi: Z -> X, from the space the generative model works in to the space the problem is posed
    in.

    This is the structure the corridor hides. In SynFlowNet, Z is sequences of reactions and X is
    molecules: the policy, the reference walk K_tau and the pool all live in Z, while the oracle,
    validity and the target region live in X. ABE already has the hook -- `query` decodes latents
    with `base_model.postprocess` before calling `validity` -- but every tensor setup so far takes
    phi = identity, which collapses the two spaces and removes the two features that make a
    reaction space hard:

      * **many-to-one.** Several routes reach one molecule. Tilting the endpoint pool then weights a
        molecule by how many cheap routes happen to reach it, which is why trajectory balance is
        needed rather than a per-step objective, and why memorylessness fails.
      * **metric mismatch.** Two molecules can be fingerprint-close while no short route joins them,
        and two routes can be one reaction apart with products that are nothing alike.

    A state cost is evaluated along a path **in Z** at states read **in X**, i.e. V(phi(z_k)). That
    composition is the point: no reward on the endpoint phi(z_T) can see it.
    """

    name: str
    fn: Callable[[torch.Tensor], torch.Tensor]
    branches: Callable[[torch.Tensor], list[torch.Tensor]]

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        return self.fn(z)

    def multiplicity(self, x: torch.Tensor) -> int:
        """How many z-branches map onto an x. 1 is a bijection; more is the DAG's non-locality."""
        return len(self.branches(x))


def identity_decoder() -> Decoder:
    """The collapse every existing tensor setup makes, kept so it can be named and compared."""
    return Decoder("identity", lambda z: z, lambda x: [x])


def fold_decoder(seam: float = 0.0) -> Decoder:
    """phi(z) = (seam - |z_0 - seam|, z_1), the simplest many-to-one map.

    Z folds about `seam` onto the half-plane x_0 <= seam, so every point of the domain has two
    preimages: two distinct routes to the same product. The two branches are *disconnected in Z* --
    a walk can only pass between them through z_0 = seam -- while being the same point in X. That
    is exactly the case where an endpoint tilt double counts and a path-aware construction does not.

    `seam` must sit beyond the domain, or the fold's image does not cover it. A fold about the
    origin cannot reach a domain that lives at negative x, which is not a quirk of the toy: a
    decoder has to be chosen compatibly with the space it is decoding into.
    """
    return Decoder(
        f"fold@{seam:g}",
        lambda z: torch.stack([seam - (z[:, 0] - seam).abs(), z[:, 1]], dim=1),
        lambda x: [x, torch.stack([2 * seam - x[:, 0], x[:, 1]], dim=1)],
    )


def shear_decoder(a: float = 1.0) -> Decoder:
    """phi(z) = (z_0 + a z_1, z_1): a bijection that distorts distance.

    One-to-one, so nothing is double counted, but a ball in Z is an ellipse in X. The reference is
    isotropic in Z and anisotropic in X, which is the milder half of the mismatch: what the model
    finds easy to reach has little to do with what the problem calls near.
    """
    return Decoder(
        f"shear{a:g}",
        lambda z: torch.stack([z[:, 0] + a * z[:, 1], z[:, 1]], dim=1),
        lambda x: [torch.stack([x[:, 0] - a * x[:, 1], x[:, 1]], dim=1)],
    )


DECODERS = {
    "identity": identity_decoder,
    "fold": fold_decoder,
    "shear": shear_decoder,
}


@dataclass(frozen=True)
class LatentDomain:
    """A domain posed in X and explored in Z through a decoder.

    Every method takes latents and answers in the problem's terms, which is the division of labour
    ABE already assumes: dispersal and the model see z, the verifier and the target see phi(z).
    """

    domain: Domain
    decoder: Decoder

    @property
    def name(self) -> str:
        return f"{self.domain.name}/{self.decoder.name}"

    def contains(self, z: torch.Tensor) -> torch.Tensor:
        return self.domain.contains(self.decoder(z))

    def in_target(self, z: torch.Tensor) -> torch.Tensor:
        return self.domain.in_target(self.decoder(z))

    def crossed(self, z: torch.Tensor) -> torch.Tensor:
        return self.domain.crossed(self.decoder(z))

    def progress(self, z: torch.Tensor) -> torch.Tensor:
        return self.domain.progress(self.decoder(z))

    def segment_valid(self, z0: torch.Tensor, z1: torch.Tensor, steps: int) -> torch.Tensor:
        """Whether the straight path in Z from z0 to z1 stays valid in X, checked at `steps`
        interior points.

        This is exactly what a hard state cost computes: the walk is in Z, the constraint is in X,
        and the answer depends on the resolution, because the reference cannot charge for an
        obstacle it never lands in. `steps = 0` sees only the endpoints, which is the
        terminal-reward view, and is blind by construction.
        """
        ok = torch.ones(len(z0), dtype=torch.bool, device=z0.device)
        for k in range(1, steps + 1):
            t = k / (steps + 1)
            ok &= self.contains(z0 + t * (z1 - z0))

        return ok
