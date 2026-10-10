from typing import Generic, Literal, Any
from dataclasses import dataclass
from contextlib import contextmanager

import numpy as np
import torch
from diffusiongym import construct_env, D, Reward
from diffusiongym.utils import train_base_model
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import logging
import yaml
import time
import csv

from .inf_methods.dps import RewardGradient
from .uncertainty import UncertaintyEstimator, FlowFeatureExtractor, uncertainty_estimators
from .setups import setups as problem_setups
from .setups.problem_setup import ProblemSetup
from .utils import write_video, filter_out_invalids, Batch, serialize_args, setup_logger


@dataclass(frozen=True)
class ExploreConfig:
    # Experiment directory
    folder: Path

    # Exploration sampling
    num_iters: int = 1000
    samples_per_iter: int = 64
    sample_batch_size: int = 64
    num_steps: int = 100

    # Feature extraction
    feat_timestep: float = 0.9
    dps_weight: float = 100.0

    # Uncertainty reward weight
    mean_weight: float = 0.0
    reward_opt: Literal["max", "min"] = "max"

    # Fine-tuning
    ft_min_dataset_size: int = 64
    ft_batch_size: int = 64
    ft_accumulate_steps: int = 1
    ft_steps: int = 500
    ft_lr: float = 1e-4
    ft_weight_decay: float = 0.0

    # Checkpointing
    ckpt_every: int = 100

    # Sampling and evaluation
    eval_samples: int = 0
    eval_batch_size: int = 64
    eval_every: int = 10
    video_fps: int = 4

    # Flags
    no_uncertainty: bool = False
    no_verifier: bool = False
    stop_when_solved: bool = False
    no_visualize: bool = False

    def __post_init__(self):
        # Create experiment directory if it doesn't exist
        self.folder.mkdir(parents=True, exist_ok=True)

        # Validation
        if not (0 <= self.feat_timestep <= 1):
            raise ValueError(f"feat_timestep must be in [0, 1], got {self.feat_timestep}")

        if self.dps_weight < 0:
            raise ValueError(f"dps_weight cannot be negative, got {self.dps_weight}")

    @staticmethod
    def construct_from_args(args: argparse.Namespace | dict) -> "ExploreConfig":
        if isinstance(args, argparse.Namespace):
            args = vars(args)

        # Map argparse flags to config names
        name_mapping = { "dir": "folder" }

        # We only take keys that exist in the dataclass fields
        config_fields = {f.name for f in ExploreConfig.__dataclass_fields__.values()}
        clean_kwargs = {}

        for key, value in args.items():
            # Check if the key needs a rename
            target_key = name_mapping.get(key, key)

            # Only add if it's a valid field in our config
            if target_key in config_fields:
                clean_kwargs[target_key] = value

        return ExploreConfig(**clean_kwargs)


class ExploreLoop(Generic[D]):
    def __init__(
        self,
        problem_setup: ProblemSetup[D],
        uncertainty: UncertaintyEstimator[D],
        reward: Reward,
        config: ExploreConfig,
        logger: logging.Logger,
    ):
        self.problem = problem_setup
        self.config = config
        self.logger = logger

        env = construct_env(
            problem_setup.base_model,
            reward,
            discretization_steps=config.num_steps,
            reward_scale=config.dps_weight,
        )

        self.base_model = problem_setup.base_model
        self.uncertainty = uncertainty
        self.env = env

        self.logger.info(f"base model parameters: {sum(p.numel() for p in self.base_model.parameters() if p.requires_grad):,}")

        self._timings = {}

        # Create file and write headers
        self.timing_heads = ["iteration", "eval", "sampling", "visualize", "finetune", "uncertainty_update", "checkpoint"]
        self.timing_path = self.config.folder / "timings.csv"
        with open(self.timing_path, "w", newline="") as f:
            csv.writer(f).writerow(self.timing_heads)

    @contextmanager
    def _timer(self, name: str):
        """Context manager to measure execution time of a block."""
        t0 = time.perf_counter()
        yield
        self._timings[name] = time.perf_counter() - t0

    def _flush_timings(self, iteration: int):
        """Writes the stored timings to the CSV file."""
        row = [iteration] + [self._timings.get(head, 0.0) for head in self.timing_heads[1:]]
        with open(self.timing_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        self._timings = {}

    def finetune_base_model(self, batches: list[Batch[D]], pbar: bool = False):
        """Fine-tune the base model with obtained valid samples.

        Parameters
        ----------
        batches : list[Batch[D]]
            Data batches obtained so far, where each entry is from a separate iteration.
        pbar : bool, default: False
            Progress bar or not.
        """

        if not self.config.no_verifier:
            valid_batch = filter_out_invalids(batches)
            if len(valid_batch) == 0:
                self.logger.warning("No valid data to fine-tune")
                return
        else:
            valid_batch = Batch.concat(batches)

        # Fine-tune base model
        opt = torch.optim.AdamW(
            self.base_model.parameters(),
            lr=self.config.ft_lr,
            weight_decay=self.config.ft_weight_decay,
        )
        train_base_model(
            self.base_model,
            opt,
            [valid_batch.latents.to(self.env.base_model.device)],
            [valid_batch.kwargs],
            steps=self.config.ft_steps,
            batch_size=self.config.ft_batch_size,
            accumulate_steps=self.config.ft_accumulate_steps,
            pbar=pbar,
        )

    def update_uncertainty_estimator(self, batches: list[Batch[D]]):
        """Update the uncertainty estimator with obtained samples.

        Parameters
        ----------
        batches : list[Batch[D]]
            Data batches obtained so far, where one batch comes from one iteration in the order.
        """
        if self.config.mean_weight > 0:
            scale = -1.0 if self.config.reward_opt == "min" else 1.0
            targets = [scale * b.rewards for b in batches]
        else:
            targets = [b.valids.float() for b in batches]

        self.uncertainty.set_data(
            [b.latents for b in batches],
            targets,
            [b.kwargs for b in batches],
        )

    def visualize_iter(self, batch: Batch[D], iteration: int):
        # Visualizing an iteration is problem-dependent
        fig = self.problem.visualize_sample(self.env, self.uncertainty, batch)

        # Save frame
        directory = self.config.folder / "frames"
        directory.mkdir(parents=True, exist_ok=True)
        fig_path = directory / f"{iteration:04d}.png"
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)

    @torch.no_grad()
    def eval_model(self, iteration: int) -> dict[str, float]:
        n = self.config.eval_samples
        bs = self.config.eval_batch_size

        if n <= 0:
            return dict()

        eval_kwargs = self.problem.eval_sampling_kwargs(n)
        sample = self.env.batch_sample(n, bs, **eval_kwargs)
        batch = Batch.from_sample(sample)
        batch.valids = self.problem.validity(batch.samples, batch.kwargs)

        # Save samples
        directory = self.config.folder / "eval" / f"{iteration:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        self.problem.save_samples(batch.samples, batch.kwargs, directory)
        torch.save(batch.valids, directory / "valids.pt")

        # Compute and save evaluation metrics
        metrics = self.problem.compute_metrics(batch.samples, batch.kwargs)
        metrics["model_valid"] = batch.valids.float().mean().item()
        with open(directory / "metrics.yaml", "w") as f:
            yaml.dump(metrics, f)

        return metrics

    def explore_loop(self, num_iterations: int, samples_per_iter: int) -> dict[str, Any]:
        metrics = dict()
        batches: list[Batch[D]] = []
        total_samples = 0
        total_valid_samples = 0
        samples_to_solve = None

        for i in range(num_iterations):
            # Evaluate current model state
            with self._timer("eval"):
                metrics = {}
                if i % self.config.eval_every == 0:
                    metrics = self.eval_model(i)

            # Determine if we should use uncertainty guidance
            has_enough_data = total_valid_samples > self.config.ft_min_dataset_size
            use_guidance = has_enough_data and not self.config.no_uncertainty

            # Collect new samples
            with self._timer("sampling"):
                if use_guidance:
                    self.env.control_policy = RewardGradient(self.env, self.uncertainty)

                sample = self.env.batch_sample(samples_per_iter, self.config.sample_batch_size)
                batch = Batch.from_sample(sample)
                batch.latents = self.problem.postprocess_latents(batch)
                batch.valids = self.problem.validity(batch.samples, batch.kwargs)
                total_samples += len(batch)
                total_valid_samples += batch.valids.int().sum().item()
                solved = self.problem.is_solved(batch)
                if solved and samples_to_solve is None:
                    samples_to_solve = total_samples

                self.env.control_policy = None

            # Store data
            batches.append(batch)

            # Logging and visualization
            with self._timer("visualize"):
                if not self.config.no_visualize:
                    self.visualize_iter(batch, i)

                with torch.no_grad():
                    mean, uncert = self.uncertainty.mean_and_uncertainty(batch.latents, **batch.kwargs)

                if self.config.mean_weight > 0:
                    metrics["pred_mean"] = mean.mean().item()
                    metrics["biased_reward"] = batch.rewards.mean().item()
                metrics["uncertainty"] = uncert.mean().item()
                metrics["biased_valid"] = batch.valids.float().mean().item()
                metrics["max_vram"] = torch.cuda.max_memory_allocated() * 1e-9
                self.logger.info(f"(iter={i:05d}) {', '.join([f'{k}: {v:.2f}' for k, v in metrics.items()])}")

            # Update models with full buffer
            with self._timer("finetune"):
                if has_enough_data:
                    self.finetune_base_model(batches, pbar=False)

            with self._timer("uncertainty_update"):
                if not self.config.no_uncertainty:
                    self.update_uncertainty_estimator(batches)

            # Save checkpoint
            with self._timer("checkpoint"):
                ckpt_dir = self.config.folder / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)

                state_dict = self.base_model.state_dict()
                torch.save(state_dict, ckpt_dir / "last.pt")

                if self.config.ckpt_every > 0 and i % self.config.ckpt_every == 0:
                    torch.save(state_dict, ckpt_dir / f"ckpt_{i}.pt")

            # Write timings to CSV
            self._flush_timings(i)

            if solved and self.config.stop_when_solved:
                self.logger.info(f"solved at iteration {i} after {total_samples} samples")
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


def setup_and_run(args: argparse.Namespace, reward: Reward, mean_weight: float):
    """Shared initialization and run logic for task-agnostic and task-directed."""
    loop = build_loop(args, reward, mean_weight)
    loop.explore_loop(loop.config.num_iters, loop.config.samples_per_iter)


def build_loop(args: argparse.Namespace, reward: Reward, mean_weight: float) -> ExploreLoop:
    """Shared initialization logic for task-agnostic and task-directed."""
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = ExploreConfig.construct_from_args(args)
    problem_setup = problem_setups[args.problem_setup](vars(args), device=device)

    # Construct feature extractor for uncertainty quantification
    feat_extractor = FlowFeatureExtractor(
        problem_setup.base_model,
        layer=problem_setup.feature_layer,
        timestep=config.feat_timestep,
        postprocess=problem_setup.postprocess_features,
    )

    # Probe the feature extractor for testing and obtaining the dimensionality of the features
    x, kwargs = problem_setup.base_model.sample_p0(1)
    x, kwargs = problem_setup.base_model.preprocess(x, **kwargs)
    feat = feat_extractor(x, **kwargs)

    if not isinstance(feat, torch.Tensor):
        raise TypeError(f"Feature extractor output must be a torch.Tensor, got {type(feat)}")

    if feat.ndim != 2:
        raise ValueError(f"Feature extractor output must be a 2D tensor, got {feat.ndim}D tensor")

    uncertainty = uncertainty_estimators[args.uncertainty_estimator](
        feat_extractor,
        feat_dim=feat.shape[1],
        mean_weight=mean_weight,
        device=device,
        args=vars(args),
    )

    # Save arguments
    with open(config.folder / "args.yaml", "w") as f:
        yaml.safe_dump(serialize_args(args), f)

    # Set up logging
    logger = setup_logger(config.folder, args.verbose)
    logger.info("starting...")

    return ExploreLoop(problem_setup=problem_setup, uncertainty=uncertainty, reward=reward, config=config, logger=logger)


def build_parser(add_extra_args):
    """Build the nested argparse parser structure.

    Parameters
    ----------
    add_extra_args : callable
        Function that adds mode-specific arguments to a parser.
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="problem_setup", required=True)

    for name, setup_cls in problem_setups.items():
        sub = subparsers.add_parser(name)

        # Create nested uncertainty subparser first
        uncertainty_subparsers = sub.add_subparsers(dest="uncertainty_estimator", required=True)

        for u_name, u_cls in uncertainty_estimators.items():
            u_sub = uncertainty_subparsers.add_parser(u_name)

            # Attach all arguments to the final nested subparser
            add_global_args(u_sub)
            add_extra_args(u_sub)
            setup_cls.add_args(u_sub)
            u_cls.add_args(u_sub)

    return parser


def add_global_args(parser):
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=None)

    # Exploration sampling
    parser.add_argument("--num_iters", type=int, default=1000)
    parser.add_argument("--samples_per_iter", type=int, default=64)
    parser.add_argument("--sample_batch_size", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=100)

    # Uncertainty estimator
    parser.add_argument("--feat_timestep", type=float, default=0.9)
    parser.add_argument("--dps_weight", type=float, default=100)

    # Fine-tuning
    parser.add_argument("--ft_min_dataset_size", type=int, default=64)
    parser.add_argument("--ft_steps", type=int, default=500)
    parser.add_argument("--ft_batch_size", type=int, default=64)
    parser.add_argument("--ft_accumulate_steps", type=int, default=1)
    parser.add_argument("--ft_lr", type=float, default=1e-4)
    parser.add_argument("--ft_weight_decay", type=float, default=0.0)

    # Checkpointing
    parser.add_argument("--ckpt_every", type=int, default=100)

    # Sampling
    parser.add_argument("--eval_samples", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--eval_every", type=int, default=10)

    # Logging
    parser.add_argument("--video_fps", type=int, default=4)
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--stop_when_solved", action="store_true")
