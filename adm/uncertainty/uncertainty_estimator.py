from typing import Generic, Callable, Any, Optional
from abc import abstractmethod

import torch
import torch.nn as nn
from diffusiongym import D, BaseModel, Reward
from diffusiongym.utils import dict_to_device
from argparse import ArgumentParser


class FlowFeatureExtractor(nn.Module, Generic[D]):
    """Makes it possible to extract features from a specific layer of a flow model.

    Parameters
    ----------
    base_model : BaseModel[D]
        The base generative flow model from which to extract features.
    layer : str
        The name of the layer from which to extract features. Use dot notation for nested layers.
        Returns the input if "input" is specified.
    timestep : float
        The timestep at which to extract features during the diffusion process. Earlier timesteps
        will give more semantic features, whereas later ones will give more low-level features.
    postprocess : Callable[[D, Any], torch.Tensor]
        An optional post-processing function to apply to the extracted features.
    """

    def __init__(
        self,
        base_model: BaseModel[D],
        layer: str,
        timestep: float,
        postprocess: Callable[[D, Any], torch.Tensor],
    ):
        super().__init__()
        self.base_model = base_model
        self.layer = layer
        self.timestep = timestep
        self._features = None
        self.postprocess = postprocess

        if layer != "input":
            self._register_hook()

    def _get_module_by_name(self, module: nn.Module, name: str):
        parts = name.split(".")
        for p in parts:
            if p.isdigit():
                module = module[int(p)]  # type: ignore
            else:
                module = getattr(module, p)

        return module

    def _register_hook(self):
        target = self._get_module_by_name(self.base_model, self.layer)

        def hook_fn(_, __, output):
            self._features = output

        self._hook = target.register_forward_hook(hook_fn)

    def forward(self, x1: D, **kwargs: Any) -> torch.Tensor:
        if self.layer == "input":
            return self.postprocess(x1, x1)

        self._features = None

        x0 = x1.randn_like()
        t = self.timestep * torch.ones(len(x1), device=x1.device)
        alpha = self.base_model.scheduler.alpha(x1, t)
        beta = self.base_model.scheduler.beta(x1, t)
        xt = alpha * x1 + beta * x0

        # Forward pass through the base model to obtain features through hook
        _ = self.base_model(xt, t, **kwargs)

        if self._features is None:
            raise RuntimeError(f"No features captured from layer '{self.layer}'")

        feats = self.postprocess(x1, self._features)
        return feats / feats.norm(dim=-1, keepdim=True)

    def remove_hook(self):
        self._hook.remove()


class UncertaintyEstimator(Reward[D]):
    """Abstract reward based on uncertainty estimates.
    
    Parameters
    ----------
    feat_extractor : FlowFeatureExtractor[D]
        Feature extractor that maps latents to features.
    mean_weight : float, default=0.0
        Weight given to the mean prediction for task-directed exploration. Set to zero for
        task-agnostic exploration.
    device : torch.device, default=cpu
        Device to update the estimator on.
    args : dict[str, Any]
        A dictionary of arguments to configure the problem setup.
    """

    def __init__(
        self,
        feat_extractor: FlowFeatureExtractor[D],
        feat_dim: int,
        mean_weight: float = 0.0,
        device: Optional[torch.device | str] = None,
        args: dict[str, Any] = {},
    ):
        if device is None:
            device = torch.device("cpu")

        self.feat_extractor = feat_extractor
        self.feat_dim = feat_dim
        self.mean_weight = mean_weight
        self.device = device
        self.args = args

        self._init_estimator()

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        """Add problem setup specific arguments to the parser.

        Parameters
        ----------
        parser : ArgumentParser
            The argument parser to which to add arguments.
        """
        pass

    @abstractmethod
    def _init_estimator(self):
        """Initialize the estimator."""
        pass

    @abstractmethod
    def _update_estimator(self, feats: torch.Tensor, labels: torch.Tensor):
        """Updates the uncertainty estimator with new features and labels.

        Parameters
        ----------
        feats : torch.Tensor
            The extracted features from the flow model.
        labels : torch.Tensor
            The corresponding labels for the features.
        """
        pass

    @abstractmethod
    def _mean_and_uncertainty(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """"Returns the mean and uncertainty for a given input.
        
        Parameters
        ----------
        feats : torch.Tensor
            The features for which to compute the mean and uncertainty.

        Returns
        -------
        mean : torch.Tensor
            The mean prediction for the input.
        uncertainty : torch.Tensor
            The uncertainty estimate for the input.
        """
        pass

    def _get_feats(self, latent: D, **kwargs: Any) -> torch.Tensor:
        latent = latent.to(self.device)
        kwargs = dict_to_device(kwargs, self.device)
        return self.feat_extractor(latent, **kwargs)

    def mean_and_uncertainty(self, latent: D, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self._get_feats(latent, **kwargs)
        return self._mean_and_uncertainty(feats)

    def set_data(
        self,
        latents: list[D],
        labels: list[torch.Tensor],
        kwargs: list[dict[str, Any]],
        standardize: bool = True,
    ):
        """Update data which is used to update the uncertainty estimator.

        Parameters
        ----------
        latents : list[D]
            Latents corresponding to batches of data points.
        labels : list[torch.Tensor]
            Labels per data point batch, which correspond to either validity in the task-agnostic case
            or reward values in the task-directed case.
        kwargs : list[dict[str, Any]]
            Keyword arguments used to obtain the latents.
        standardize : bool, default: True
            Whether to standardize the labels to zero mean and unit variance. With imbalanced binary
            labels, this makes the minority class weigh much more than the majority class.
        """
        with torch.no_grad():
            feats = [self._get_feats(x, **kw) for x, kw in zip(latents, kwargs)]

        feats_tensor = torch.cat(feats, dim=0)
        labels_tensor = torch.cat(labels, dim=0)

        if standardize:
            labels_tensor = (labels_tensor - labels_tensor.mean()) / (labels_tensor.std() + 1e-8)

        labels_tensor = labels_tensor.to(self.device)

        self._update_estimator(feats_tensor, labels_tensor)

    def __call__(self, sample: D | None, latent: D, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        mean, uncertainty = self.mean_and_uncertainty(latent, **kwargs)
        return self.mean_weight * mean + uncertainty, torch.ones(len(latent), dtype=torch.bool)
