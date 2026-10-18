from typing import Generic, Any, Callable, Optional
from abc import ABC, abstractmethod

import torch
from diffusiongym import D, BaseModel, Environment, DDTensor
from matplotlib.figure import Figure
from pathlib import Path
from argparse import ArgumentParser

from adm.uncertainty import UncertaintyEstimator
from adm.utils import Batch


class ProblemSetup(ABC, Generic[D]):
    def __init__(self, args: dict[str, Any]):
        """Initialize the problem setup with given arguments.

        Parameters
        ----------
        args : dict[str, Any]
            A dictionary of arguments to configure the problem setup.
        """
        pass

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        """Add problem setup specific arguments to the parser.

        Parameters
        ----------
        parser : ArgumentParser
            The argument parser to which to add arguments.
        """
        pass

    @property
    @abstractmethod
    def base_model(self) -> BaseModel[D]:
        """The (data pre-trained) generative flow model used in the problem setup."""
        raise NotImplementedError

    @abstractmethod
    def validity(self, samples: D, kwargs: dict[str, Any]) -> torch.Tensor:
        """Validity/verifier function that checks whether a sample is valid or not.

        Parameters
        ----------
        samples : D
            The samples to check.
        kwargs : dict
            The keyword arguments used to generate the samples.

        Returns
        -------
        torch.Tensor
            A boolean tensor indicating whether each sample is valid.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def feature_layer(self) -> str:
        """The name of the layer from which to extract features for the GP."""
        raise NotImplementedError

    def is_solved(self, batch: Batch[D]) -> bool:
        """Whether an exploration batch has solved the problem, e.g., reached a target region. Only
        used to stop exploration early when `stop_when_solved` is set.

        Parameters
        ----------
        batch : Batch[D]
            The sample.
        Returns
        -------
        bool
            Whether the problem is solved.
        """
        return False

    def disperse(
        self,
        env: Environment[D],
        latents: D,
        scale: float,
        noise: Callable[[int, int], torch.Tensor],
    ) -> D:
        """Perturb latents locally, used by ABE to disperse its pool under the reference process.

        Parameters
        ----------
        env : Environment[D]
            The environment of the current model, e.g., to regenerate partially noised latents.
        latents : D
            The latents to perturb.
        scale : float
            The scale of the perturbation.
        noise : Callable[[int, int], torch.Tensor]
            Samples n displacements of dimension d with the given scale under ABE's reference law.

        Returns
        -------
        D
            The perturbed latents.

        Notes
        -----
        The default adds `noise` to tensor latents. Other latents, e.g., molecular graphs, have no
        meaningful sum with a displacement and must override this method, for example by noising the
        latents to a time that depends on `scale` and regenerating them with the model.
        """
        if not isinstance(latents, DDTensor):
            raise NotImplementedError(f"{type(self).__name__} must implement `disperse` for {type(latents).__name__} latents")

        disp = noise(len(latents), latents.data[0].numel()).to(latents.data.device)
        return DDTensor(latents.data + disp.reshape(latents.data.shape))  # type: ignore

    def distance_features(self, latents: D) -> Optional[torch.Tensor]:
        """Features in which ABE measures distances between latents, for diverse selection and
        for its trust region.

        Parameters
        ----------
        latents : D
            The latents.

        Returns
        -------
        torch.Tensor | None
            A tensor of shape (n, d), or None to use the features of the uncertainty estimator.

        Notes
        -----
        The default flattens tensor latents and returns None otherwise. Features should not depend on
        arbitrary choices in the representation, such as the order or orientation of atoms.
        """
        if isinstance(latents, DDTensor):
            return latents.data.flatten(1)

        return None

    def distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Pairwise distances between two sets of distance features.

        Parameters
        ----------
        a : torch.Tensor, shape (n, d)
            The first set of features.
        b : torch.Tensor, shape (m, d)
            The second set of features.

        Returns
        -------
        torch.Tensor, shape (n, m)
            The pairwise distances. The default is Euclidean.
        """
        return torch.cdist(a, b)

    def postprocess_latents(self, batch: Batch[D]) -> D:
        """Post-process latents generated from the base model.

        Parameters
        ----------
        batch : Batch[D]
            The sample.

        Returns
        -------
        latent : D
            The post-processed latents.
        """
        return batch.latents

    @abstractmethod
    def postprocess_features(self, latents: D, feats: Any) -> torch.Tensor:
        """Post-process features extracted from the base model.

        Parameters
        ----------
        latents : D
            The batch.
        feats : Any
            The raw features extracted from the base model.

        Returns
        -------
        features : torch.Tensor
            The post-processed features.
        """
        raise NotImplementedError

    @abstractmethod
    def visualize_sample(self, env: Environment[D], uncertainty: UncertaintyEstimator[D], batch: Batch[D]) -> Figure:
        """Produce a matplotlib figure for visualizing the sample in the problem setup.

        Parameters
        ----------
        env : Environment[D]
            The environment in which the samples were generated.
        uncertainty : UncertaintyEstimator[D]
            Current uncertainty estimator.
        batch : Batch[D]
            The batch.
        """
        raise NotImplementedError

    @abstractmethod
    def save_samples(self, samples: D, kwargs: dict, dir: Path) -> bool:
        """Save a batch of samples to the disk.

        Parameters
        ----------
        samples : D
            The samples to save.
        kwargs : dict
            The keyword arguments used to generate the samples.
        dir : Path
            The directory where to save the samples, without extension.

        Returns
        -------            
        bool
            Whether the samples were saved successfully or not.

        Notes
        -----
        You do not need to save all keyword arguments, only the ones that might be necessary for
        evalation. For example, for Stable Diffusion, the encoder_hidden_states are not important,
        but the prompts are.
        """
        raise NotImplementedError

    @abstractmethod
    def load_samples(self, dir: Path) -> tuple[D, dict]:
        """Load a batch of samples from the disk, saved by `save_samples`.

        Parameters
        ----------
        dir : Path
            The directory from which to load the samples.

        Returns
        -------
        tuple[D, dict] 
            A tuple of the loaded samples and their corresponding keyword arguments.
        """
        raise NotImplementedError

    def eval_sampling_kwargs(self, n: int) -> dict[str, Any]:
        """Provide keyword arguments for sampling during evaluation.

        Parameters
        ----------
        n : int
            The number of samples to be drawn.

        Returns
        -------
        dict[str, Any]
            A dictionary of keyword arguments for sampling.
        """
        return {}

    def compute_metrics(self, samples: D, kwargs: dict) -> dict[str, float]:
        """Compute global metrics for the problem setup.
        
        Parameters
        ----------
        samples : D
            The samples to compute metrics on.
        kwargs : dict
            The keyword arguments used to generate the samples.

        Returns
        -------
        dict[str, float]
            A dictionary of computed metrics.
        """
        return dict()

    def compute_sample_metrics(self, samples: D, kwargs: dict) -> list[dict[str, float]]:
        """Compute relevant metrics on individual samples.

        Parameters
        ----------
        samples : D
            The samples to compute metrics on.
        kwargs : dict
            The keyword arguments used to generate the samples.

        Returns
        -------
        list[dict[str, float]]
            A list of dictionaries of computed metrics for each sample.
        """
        return [{} for _ in range(len(samples))]
