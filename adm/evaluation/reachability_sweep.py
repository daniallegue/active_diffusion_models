"""ActFlow vs ABE vs ABE with a path cost, on domains where the route and the shortcut disagree.

`corridor_sweep` sweeps (width, length) and fits the cost of crossing a corridor. That grid cannot
test a path cost at all: in the corridor the valid route *is* the straight line, so detour_ratio is
exactly 1 and no state cost has anything to discriminate. This sweep varies the **arm** instead, on
a fixed geometry, and runs each arm on domains whose detour ratio is dialled up.

The arms, in increasing order of what they are allowed to know:

    actflow      No reference process at all: exploration is DPS guidance on GP uncertainty inside
                 the sampler. There is nowhere to put a path cost, which is the point of including
                 it -- it is the baseline that can only ever see endpoints.
    abe          Dispersal in a single displacement, then tilt. The original line 4.
    abe-steps    The same dispersal split into `disperse_steps` increments, with no cost. A control,
                 and a necessary one: without it any difference could be attributed to simulating
                 the reference differently rather than to charging for the path.
    abe-soft     V = -log P(valid) along the path, from the surrogate.
    abe-hard     The same, and a path crossing mu + lambda sigma < 0 is pruned outright.

What separates them is `final_frac_shortcut`: mass the final model puts in the forbidden band
between S_0 and A, which can only have arrived by crossing. The prediction is that actflow and abe
cross freely, abe-steps behaves like abe, and abe-hard does not cross but pays for it in
`samples_to_solve`. It is a trade, not a free win, and the sweep measures its size.

The resolution of each arm is read off the domain rather than guessed: `required_steps()` inverts
the detection rate min(1, n b / d) for that domain's wall, so a thin wall automatically gets more
increments. A domain with no wall gets 1, since no resolution helps there.
"""

from typing import Any
import argparse
import json
import time
from pathlib import Path

import torch
from diffusiongym import DummyReward

from ..abe import ABELoop, ABEConfig, add_abe_args
from ..explore import build_loop, add_global_args
from ..setups import domains
from ..setups.corridor import CorridorProblemSetup
from ..uncertainty import uncertainty_estimators


# Each arm is a set of overrides applied on top of the command line. `steps` is resolved per domain.
ARMS: dict[str, dict[str, Any]] = {
    "actflow": {"method": "actflow"},
    "abe": {"method": "abe", "abe_disperse_steps": 1, "abe_state_cost": "none"},
    "abe-steps": {"method": "abe", "abe_disperse_steps": "auto", "abe_state_cost": "none"},
    "abe-soft": {"method": "abe", "abe_disperse_steps": "auto", "abe_state_cost": "surrogate",
                 "abe_state_cost_hard": False},
    "abe-hard": {"method": "abe", "abe_disperse_steps": "auto", "abe_state_cost": "surrogate",
                 "abe_state_cost_hard": True},
}


# The ablation of reports/reachability/degenerate_fixes.md. Each arm adds one fix to the arm it is
# built from, so the step between two adjacent arms is attributable; `+all` is every fix at once.
_HARD, _SOFT, _STEPS = ARMS["abe-hard"], ARMS["abe-soft"], ARMS["abe-steps"]
_ORACLE = {**_HARD, "abe_state_cost": "oracle"}
_MASS = {"abe_positive_mass": 0.0}
_SELECT = {"abe_select_path_weight": True}
_LINEAGE = {"abe_lineage": True}
_ESS = {"abe_ess_over_alive": True}
_EXCESS = {"abe_state_cost_form": "excess"}

ABLATION: dict[str, dict[str, Any]] = {
    "abe-oracle": _ORACLE,                                              # 2: upper bound
    "abe-steps+mass": {**_STEPS, **_MASS},                              # control under fix 3
    "abe-hard+mass": {**_HARD, **_MASS},                                # 3
    "abe-soft+mass": {**_SOFT, **_MASS},                                # 3
    "abe-soft+mass+excess": {**_SOFT, **_MASS, **_EXCESS},              # 4
    "abe-hard+mass+select": {**_HARD, **_MASS, **_SELECT},              # 5
    "abe-hard+mass+select+lineage": {**_HARD, **_MASS, **_SELECT, **_LINEAGE},  # 6
    "abe-hard+all": {**_HARD, **_MASS, **_SELECT, **_LINEAGE, **_ESS},  # + minor
    "abe-soft+all": {**_SOFT, **_MASS, **_EXCESS, **_SELECT, **_LINEAGE, **_ESS},
    "abe-oracle+all": {**_ORACLE, **_MASS, **_SELECT, **_LINEAGE, **_ESS},
}
ALL_ARMS = {**ARMS, **ABLATION}


def domain_for(args: argparse.Namespace, family: str) -> domains.Domain:
    """The domain a cell will run on, used for its resolution and for reporting its detour ratio."""
    return domains.build(
        family,
        width=args.corridor_width,
        length=args.corridor_length,
        gap=args.corridor_gap,
        island=args.corridor_island_size,
        x0=args.corridor_x0,
        folds=args.switchback_folds,
    )


def geometry_args(family: str) -> dict[str, Any]:
    """How a family is expressed to `CorridorProblemSetup`, which knows `corridor` and `hook`.

    `islands` is the corridor with no channel. `switchback` has no setup support yet, so it is
    rejected here rather than silently run as something else.
    """
    if family == "islands":
        return {"corridor_geometry": "corridor", "corridor_width": 0.0}

    if family in ("corridor", "hook"):
        return {"corridor_geometry": family}

    raise ValueError(
        f"{family!r} exists in adm.setups.domains but CorridorProblemSetup cannot build it yet; "
        "families are limited to corridor, islands and hook until the setup is pointed at a Domain"
    )


def cell_name(family: str, arm: str, seed: int) -> str:
    return f"{family}_{arm}_seed{seed}"


def grid(args: argparse.Namespace) -> list[tuple[str, str, int]]:
    return [(f, a, s) for f in args.families for a in args.arms for s in args.seeds]


def run_cell(args: argparse.Namespace, family: str, arm: str, seed: int):
    out = args.dir / "results" / f"{cell_name(family, arm, seed)}.json"
    if out.exists():
        print(f"skipping {out.name}, already done")
        return

    domain = domain_for(args, family)
    overrides = dict(ALL_ARMS[arm])
    if overrides.get("abe_disperse_steps") == "auto":
        # A domain with no wall reports 1, since no resolution helps there. A cost arm still needs
        # at least 2, because a single displacement has no intermediate state to charge and the
        # config rejects that combination -- and running the cost on the corridor is the point of
        # the control, so it must not be skipped.
        floor = 2 if overrides.get("abe_state_cost", "none") != "none" else 1
        overrides["abe_disperse_steps"] = max(floor, domain.required_steps(args.detect))

    cell_args = argparse.Namespace(**{
        **vars(args),
        **geometry_args(family),
        **overrides,
        "problem_setup": "corridor",
        "dir": args.dir / "runs" / cell_name(family, arm, seed),
        "seed": seed,
        # S_0 does not depend on the family, so one pre-trained model serves every cell of a seed
        "corridor_base_ckpt": str(args.dir / "base_models" / f"seed{seed}.pt"),
        # Never stop at the first hit. This sweep is decided on where the *final model's* support
        # ends up, so a run that halts the moment one query lands in A reports metrics for a model
        # that has barely trained -- and every arm hits A early here, which collapsed the first
        # attempt to 2-10 rounds. `samples_to_solve` is still recorded, since the loop notes it
        # whether or not it stops.
        "stop_when_solved": False,
    })

    method = overrides["method"]
    t0 = time.perf_counter()
    if method == "abe":
        loop = build_loop(cell_args, reward=DummyReward(), mean_weight=0, loop_cls=ABELoop,
                          abe_config=ABEConfig.construct_from_args(cell_args))
    else:
        loop = build_loop(cell_args, reward=DummyReward(), mean_weight=0)

    summary = loop.explore_loop(loop.config.num_iters, loop.config.samples_per_iter)
    runtime = time.perf_counter() - t0

    with torch.no_grad():
        sample = loop.env.batch_sample(args.final_eval_samples, args.eval_batch_size).sample

    metrics = loop.problem.compute_metrics(sample, {})
    per_round = args.abe_query_budget if method == "abe" else loop.config.samples_per_iter

    result = {
        "family": family,
        "arm": arm,
        "method": method,
        "seed": seed,
        "disperse_steps": overrides.get("abe_disperse_steps", 1),
        "detour_ratio": domain.detour_ratio,
        "wall_thickness": domain.wall_thickness(),
        "budget": loop.config.num_iters * per_round,
        "runtime": runtime,
        **summary,
        **{f"final_{k}": v for k, v in metrics.items()},
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result))


def load_results(dir: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted((dir / "results").glob("*.json"))]


def summarise(dir: Path):
    """Prints the comparison: who solved, how fast, and how much of the model crossed."""
    rows = load_results(dir)
    if not rows:
        print("no results yet")
        return

    def med(xs: list[float]) -> float:
        xs = sorted(xs)
        n = len(xs)
        return float("nan") if n == 0 else (xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2]))

    families = sorted({r["family"] for r in rows})
    order = [a for a in ALL_ARMS if any(r["arm"] == a for r in rows)]

    for family in families:
        sub = [r for r in rows if r["family"] == family]
        ratio = sub[0]["detour_ratio"]
        print(f"\n{family}  (detour_ratio {ratio:.2f}, wall {sub[0]['wall_thickness']:.3f})")
        width = max(10, *(len(a) for a in order))
        print(f"  {'arm':<{width}} {'steps':>5} {'solved':>7} {'queries':>9} {'clean':>7} {'shortcut':>9} "
              f"{'in A':>7} {'muCrossA':>8} {'A cross':>7} {'A dirty':>8}")
        for arm in order:
            cells = [r for r in sub if r["arm"] == arm]
            if not cells:
                continue

            solved = [r for r in cells if r["solved"]]
            q = med([r["samples_to_solve"] for r in solved]) if solved else float("nan")
            clean = [r["samples_to_solve_clean"] for r in cells if r.get("samples_to_solve_clean") is not None]
            print(f"  {arm:<{width}} {cells[0]['disperse_steps']:>5} "
                  f"{len(solved)}/{len(cells):<5} "
                  f"{q:>9.0f} "
                  f"{med(clean) if clean else float('nan'):>7.0f} "
                  f"{med([r['final_frac_shortcut'] for r in cells]):>9.4f} "
                  f"{med([r['final_frac_in_target'] for r in cells]):>7.4f} "
                  f"{med([path_metric(r, 'mu_cross_A') for r in cells]):>8.4f} "
                  f"{med([share(r, 'positives_A_cross') for r in cells]):>7.3f} "
                  f"{med([share(r, 'positives_A_dirty') for r in cells]):>8.3f}")

    print("\n`shortcut` is mass the final model puts in the forbidden band between S_0 and A.")
    print("`clean` is queries to the first positive in A whose ancestry never crossed the band.")
    print("`muCrossA` is the mean over rounds of mu* mass on endpoints in A reached by crossing the band;")
    print("`A cross` / `A dirty` are the shares of positives in A whose own path / any ancestor crossed.")


def path_metric(row: dict[str, Any], key: str) -> float:
    """Mean over rounds of a per-round path diagnostic, or nan for cells that predate them."""
    xs = [r[key] for r in row.get("rounds", []) if key in r]
    return sum(xs) / len(xs) if xs else float("nan")


def share(row: dict[str, Any], key: str) -> float:
    total = row.get("positives_A", 0)
    return row.get(key, float("nan")) / total if total else float("nan")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    subparsers = parser.add_subparsers(dest="uncertainty_estimator", required=True)

    for u_name, u_cls in uncertainty_estimators.items():
        sub = subparsers.add_parser(u_name)
        add_global_args(sub)
        CorridorProblemSetup.add_args(sub)
        u_cls.add_args(sub)

        sub.add_argument("--no_uncertainty", action="store_true")
        sub.add_argument("--no_verifier", action="store_true")
        add_abe_args(sub)

        sub.add_argument("--families", type=str, nargs="+", default=["corridor", "hook"],
                         help="corridor is the control: its detour ratio is exactly 1")
        sub.add_argument("--arms", type=str, nargs="+", default=list(ARMS),
                         choices=list(ALL_ARMS))
        sub.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
        sub.add_argument("--switchback_folds", type=int, default=3,
                         help="Only read when a switchback family is requested")
        sub.add_argument("--detect", type=float, default=0.9,
                         help="Wall-detection probability the automatic disperse_steps targets")
        sub.add_argument("--grid_index", type=int, default=None,
                         help="Run only this cell (e.g. a SLURM array id)")
        sub.add_argument("--final_eval_samples", type=int, default=2048)
        sub.add_argument("--summary_only", action="store_true")

    return parser


def main(args: argparse.Namespace):
    if not args.summary_only:
        cells = grid(args)
        if args.grid_index is not None:
            if args.grid_index >= len(cells):
                raise SystemExit(f"grid_index {args.grid_index} is past the last cell ({len(cells) - 1})")

            cells = [cells[args.grid_index]]

        for family, arm, seed in cells:
            run_cell(args, family, arm, seed)

    if args.grid_index is None:
        summarise(args.dir)


if __name__ == "__main__":
    main(build_parser().parse_args())
