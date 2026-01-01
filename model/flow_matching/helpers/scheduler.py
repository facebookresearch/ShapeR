# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.
#
# This file is derived from https://github.com/facebookresearch/flow_matching
# Licensed under CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/

# pyre-unsafe

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.distributions import LogisticNormal, Uniform


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


@dataclass
class SchedulerOutput:
    r"""Represents a sample of a conditional-flow generated probability path.

    Attributes:
        alpha_t (Tensor): :math:`\alpha_t`, shape (...).
        sigma_t (Tensor): :math:`\sigma_t`, shape (...).
        d_alpha_t (Tensor): :math:`\frac{\partial}{\partial t}\alpha_t`, shape (...).
        d_sigma_t (Tensor): :math:`\frac{\partial}{\partial t}\sigma_t`, shape (...).

    """

    alpha_t: Tensor = field(metadata={"help": "alpha_t"})
    sigma_t: Tensor = field(metadata={"help": "sigma_t"})
    d_alpha_t: Tensor = field(metadata={"help": "Derivative of alpha_t."})
    d_sigma_t: Tensor = field(metadata={"help": "Derivative of sigma_t."})


class CondOTScheduler:
    """CondOT Scheduler."""

    def __call__(self, t: Tensor) -> SchedulerOutput:
        r"""
        Args:
            t (Tensor): times in [0,1], shape (...).

        Returns:
            SchedulerOutput: :math:`\alpha_t,\sigma_t,\frac{\partial}{\partial t}\alpha_t,\frac{\partial}{\partial t}\sigma_t`
        """
        return SchedulerOutput(
            alpha_t=t,
            sigma_t=1 - t,
            d_alpha_t=torch.ones_like(t),
            d_sigma_t=-torch.ones_like(t),
        )

    def snr_inverse(self, snr: Tensor) -> Tensor:
        r"""
        Computes :math:`t` from the signal-to-noise ratio :math:`\frac{\alpha_t}{\sigma_t}`.

        Args:
            snr (Tensor): The signal-to-noise, shape (...)

        Returns:
            Tensor: t, shape (...)
        """
        kappa_t = snr / (1.0 + snr)

        return self.kappa_inverse(kappa=kappa_t)

    def kappa_inverse(self, kappa: Tensor) -> Tensor:
        """
        Computes :math:`t` from :math:`kappa_t`.

        Args:
            kappa (Tensor): :math:`kappa`, shape (...)

        Returns:
            Tensor: t, shape (...)
        """
        return kappa


class TimeSampler:
    def __init__(self, dist):
        self.dist_name = dist
        if dist == "uniform":
            self.distribution = Uniform(0.0, 1.0)
        elif dist == "lognormal":
            self.distribution = LogisticNormal(0.0, 1.0)
        elif dist == "flux":
            self.distribution = FluxTimeSampler(mode="train")
        else:
            raise ValueError(f"Unknown distribution {dist}")

    def __call__(self, shape, num_tokens, device):
        if self.dist_name == "uniform":
            return self.distribution.sample(sample_shape=shape).to(device)
        elif self.dist_name == "lognormal":
            return self.distribution.sample(sample_shape=shape)[:, 0].to(device)
        elif self.dist_name == "flux":
            return self.distribution(shape, num_tokens, device)


class FluxTimeSampler:
    def __init__(self, **kwargs):
        self.mode = kwargs.get("mode", "train")

    def __call__(self, bs, num_tokens, device):
        if self.mode == "train":
            logits_norm = torch.randn(bs, device=device)
            timesteps = logits_norm.sigmoid()
            mu = get_lin_function(y1=0.5, y2=1.15)(num_tokens)
            timesteps = time_shift(mu, 1.0, timesteps)
            timesteps = 1 - timesteps
        elif self.mode == "inference":
            timesteps = torch.linspace(1, 0, bs + 1).to(device)
            mu = get_lin_function(y1=0.5, y2=1.15)(num_tokens)
            timesteps = time_shift(mu, 1.0, timesteps)
            timesteps = 1 - timesteps
        return timesteps


def visualize_sampling():
    import matplotlib.pyplot as plt

    flux_sampler = FluxTimeSampler(mode="inference")
    timesteps = flux_sampler(25, 2048, torch.device("cuda"))
    x = torch.linspace(0, 1, 25 + 1)
    plt.scatter(x, y=timesteps.cpu().numpy())
    plt.savefig("scatter_plot.png")

    flux_sampler = FluxTimeSampler(mode="train")
    timesteps = flux_sampler(5000, 4096, torch.device("cuda"))
    plt.hist(timesteps.cpu().numpy(), bins=50, edgecolor="black")
    plt.savefig("scatter_plot_4096.png")
    print(timesteps)
