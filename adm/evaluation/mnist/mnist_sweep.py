from typing import Any
import argparse
import json
import time
from pathlib import Path

import torch
from diffusiongym import DummyReward

from ...abe import ABELoop, ABEConfig, add_abe_args
from ...explore import build_loop, add_global_args
from ...setups.mnist_latent import MNISTLatentProblemSetup
from ...uncertainty import uncertainty_estimators
from ..reachability_sweep import SCALE


ARMS: dict[str, dict[str, Any]] = {
    "actflow": {"method": "actflow"},
    "actflow-nodps": {"method": "actflow", "no_uncertainty": True},
    "actflow@dps100": {"method": "actflow", "dps_weight": 100.0},
    "actflow@dps300": {"method": "actflow", "dps_weight": 300.0},
    **{arm: SCALE[arm] for arm in ("abe+mass@gaussian", "abe+mass@levy", "abe+mass@student")},
}


def grid(args: argparse.Namespace) -> list[tuple[str, int]]:
    return [(a, s) for a in args.arms for s in args.seeds]


def base_ckpt(args: argparse.Namespace, seed: int) -> Path:
    return args.dir / "base_models" / f"seed{seed}.pt"


def pretrain(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in args.seeds:
        ckpt = base_ckpt(args, seed)
        if ckpt.exists():
            print(f"skipping {ckpt.name}, already trained")
            continue

        torch.manual_seed(seed)
        MNISTLatentProblemSetup({**vars(args), "mnist_base_ckpt": str(ckpt)}, device=device)
        print(f"wrote {ckpt}")


def run_cell(args: argparse.Namespace, arm: str, seed: int):
    name = f"{arm}_seed{seed}"
    out = args.dir / "results" / f"{name}.json"
    if out.exists():
        print(f"skipping {out.name}, already done")
        return

    overrides = dict(ARMS[arm])
    method = overrides["method"]
    per_round = args.abe_query_budget if method == "abe" else args.samples_per_iter
    if method == "actflow" and args.equal_budget:
        # ActFlow spends samples_per_iter queries a round and ABE abe_query_budget, so the same
        # num_iters would hand them different budgets
        overrides["num_iters"] = max(1, round(args.num_iters * args.abe_query_budget / args.samples_per_iter))

    run_dir = args.dir / "runs" / name
    cell_args = argparse.Namespace(**{
        **vars(args),
        **overrides,
        "problem_setup": "mnist_latent",
        "dir": run_dir,
        "seed": seed,
        "mnist_base_ckpt": str(base_ckpt(args, seed)),
        # Snapshots of the model at the same query counts for every arm, whatever its round size
        "eval_every": max(1, args.eval_queries // per_round),
        "stop_when_solved": False,
    })

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

    problem: MNISTLatentProblemSetup = loop.problem
    torch.save(sample.data.cpu(), run_dir / "final_samples.pt")
    problem.save_trace(run_dir / "trace.pt")

    result = {
        "arm": arm,
        "method": method,
        "seed": seed,
        "tau": problem.tau,
        "seed_classes": problem.seed_classes,
        "dps_weight": cell_args.dps_weight if method == "actflow" and not cell_args.no_uncertainty else None,
        "reference": cell_args.abe_reference if method == "abe" else None,
        "per_round": per_round,
        "budget": loop.config.num_iters * per_round,
        "eval_every": cell_args.eval_every,
        "runtime": runtime,
        **summary,
        **problem.hit_summary(),
        **{f"final_{k}": v for k, v in problem.compute_metrics(sample, {}).items()},
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps({k: v for k, v in result.items() if k not in ("islands", "rounds")}))


def summarise(dir: Path):
    rows = [json.loads(p.read_text()) for p in sorted((dir / "results").glob("*.json"))]
    if not rows:
        print("no results yet")
        return

    print(f"\n{'arm':<20} {'seeds':>5} {'islands hit':>12} {'new classes':>12} {'first hit':>10} "
          f"{'mass on islands':>16}")
    for arm in [a for a in ARMS if any(r["arm"] == a for r in rows)]:
        cells = [r for r in rows if r["arm"] == arm]
        hits = sorted(r["islands_hit"] for r in cells)
        classes = sorted(len(r["classes_hit"]) for r in cells)
        first = sorted(r["samples_to_solve"] for r in cells if r["solved"])
        mass = sorted(r["final_frac_in_target"] for r in cells)
        print(f"{arm:<20} {len(cells):>5} {hits[len(hits) // 2]:>5}/{cells[0]['num_islands']:<6} "
              f"{classes[len(classes) // 2]:>5}/{cells[0]['num_new_classes']:<6} "
              f"{first[len(first) // 2] if first else float('nan'):>10} {mass[len(mass) // 2]:>16.4f}")
    print("\nmedians over seeds; `first hit` is queries to the first island, over the seeds that hit one")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    subparsers = parser.add_subparsers(dest="uncertainty_estimator", required=True)

    for u_name, u_cls in uncertainty_estimators.items():
        sub = subparsers.add_parser(u_name)
        add_global_args(sub)
        MNISTLatentProblemSetup.add_args(sub)
        u_cls.add_args(sub)
        add_abe_args(sub)

        sub.add_argument("--no_uncertainty", action="store_true")
        sub.add_argument("--no_verifier", action="store_true")
        sub.add_argument("--arms", type=str, nargs="+", default=list(ARMS), choices=list(ARMS))
        sub.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
        sub.add_argument("--equal_budget", action="store_true",
                         help="Scale ActFlow's num_iters so it spends as many queries as ABE")
        sub.add_argument("--eval_queries", type=int, default=320,
                         help="Snapshot the model's samples every this many queries")
        sub.add_argument("--pretrain_only", action="store_true")
        sub.add_argument("--grid_index", type=int, default=None, help="Run only this cell")
        sub.add_argument("--final_eval_samples", type=int, default=2048)
        sub.add_argument("--summary_only", action="store_true")

    return parser


def main(args: argparse.Namespace):
    if args.pretrain_only:
        pretrain(args)
        return

    if not args.summary_only:
        cells = grid(args)
        if args.grid_index is not None:
            if args.grid_index >= len(cells):
                raise SystemExit(f"grid_index {args.grid_index} is past the last cell ({len(cells) - 1})")

            cells = [cells[args.grid_index]]

        for arm, seed in cells:
            run_cell(args, arm, seed)

    if args.grid_index is None:
        summarise(args.dir)


if __name__ == "__main__":
    main(build_parser().parse_args())
