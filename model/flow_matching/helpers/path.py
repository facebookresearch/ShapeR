# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.
#
# This file is derived from https://github.com/facebookresearch/flow_matching
# Licensed under CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/

# pyre-unsafe

from dataclasses import dataclass, field

from torch import Tensor


def expand_tensor_like(input_tensor: Tensor, expand_to: Tensor) -> Tensor:
    """`input_tensor` is a 1d vector of length equal to the batch size of `expand_to`,
    expand `input_tensor` to have the same shape as `expand_to` along all remaining dimensions.

    Args:
        input_tensor (Tensor): (batch_size,).
        expand_to (Tensor): (batch_size, ...).

    Returns:
        Tensor: (batch_size, ...).
    """
    assert input_tensor.ndim == 1, "Input tensor must be a 1d vector."
    assert (
        input_tensor.shape[0] == expand_to.shape[0]
    ), f"The first (batch_size) dimension must match. Got shape {input_tensor.shape} and {expand_to.shape}."

    dim_diff = expand_to.ndim - input_tensor.ndim

    t_expanded = input_tensor.clone()
    t_expanded = t_expanded.reshape(-1, *([1] * dim_diff))

    return t_expanded.expand_as(expand_to)


@dataclass
class PathSample:
    r"""Represents a sample of a conditional-flow generated probability path.

    Attributes:
        x_1 (Tensor): the target sample :math:`X_1`.
        x_0 (Tensor): the source sample :math:`X_0`.
        t (Tensor): the time sample :math:`t`.
        x_t (Tensor): samples :math:`X_t \sim p_t(X_t)`, shape (batch_size, ...).
        dx_t (Tensor): conditional target :math:`\frac{\partial X}{\partial t}`, shape: (batch_size, ...).

    """

    x_1: Tensor = field(metadata={"help": "target samples X_1 (batch_size, ...)."})
    x_0: Tensor = field(metadata={"help": "source samples X_0 (batch_size, ...)."})
    t: Tensor = field(metadata={"help": "time samples t (batch_size, ...)."})
    x_t: Tensor = field(
        metadata={"help": "samples x_t ~ p_t(X_t), shape (batch_size, ...)."}
    )
    dx_t: Tensor = field(
        metadata={"help": "conditional target dX_t, shape: (batch_size, ...)."}
    )


class AffineProbPath:
    r"""The ``AffineProbPath`` class represents a specific type of probability path where the transformation between distributions is affine.
    An affine transformation can be represented as:

    .. math::

        X_t = \alpha_t X_1 + \sigma_t X_0,

    where :math:`X_t` is the transformed data point at time `t`. :math:`X_0` and :math:`X_1` are the source and target data points, respectively. :math:`\alpha_t` and :math:`\sigma_t` are the parameters of the affine transformation at time `t`.

    The scheduler is responsible for providing the time-dependent parameters :math:`\alpha_t` and :math:`\sigma_t`, as well as their derivatives, which define the affine transformation at any given time `t`.

    Using ``AffineProbPath`` in the flow matching framework:

    .. code-block:: python

        # Instantiates a probability path
        my_path = AffineProbPath(...)
        mse_loss = torch.nn.MSELoss()

        for x_1 in dataset:
            # Sets x_0 to random noise
            x_0 = torch.randn()

            # Sets t to a random value in [0,1]
            t = torch.rand()

            # Samples the conditional path X_t ~ p_t(X_t|X_0,X_1)
            path_sample = my_path.sample(x_0=x_0, x_1=x_1, t=t)

            # Computes the MSE loss w.r.t. the velocity
            loss = mse_loss(path_sample.dx_t, my_model(x_t, t))
            loss.backward()

    Args:
        scheduler (Scheduler): An instance of a scheduler that provides the parameters :math:`\alpha_t`, :math:`\sigma_t`, and their derivatives over time.

    """

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def sample(self, x_0: Tensor, x_1: Tensor, t: Tensor):
        r"""Sample from the affine probability path:

        | given :math:`(X_0,X_1) \sim \pi(X_0,X_1)` and a scheduler :math:`(\alpha_t,\sigma_t)`.
        | return :math:`X_0, X_1, X_t = \alpha_t X_1 + \sigma_t X_0`, and the conditional velocity at :math:`X_t, \dot{X}_t = \dot{\alpha}_t X_1 + \dot{\sigma}_t X_0`.

        Args:
            x_0 (Tensor): source data point, shape (batch_size, ...).
            x_1 (Tensor): target data point, shape (batch_size, ...).
            t (Tensor): times in [0,1], shape (batch_size).

        Returns:
            PathSample: a conditional sample at :math:`X_t \sim p_t`.
        """
        scheduler_output = self.scheduler(t)
        alpha_t = expand_tensor_like(
            input_tensor=scheduler_output.alpha_t, expand_to=x_1
        )
        sigma_t = expand_tensor_like(
            input_tensor=scheduler_output.sigma_t, expand_to=x_1
        )
        d_alpha_t = expand_tensor_like(
            input_tensor=scheduler_output.d_alpha_t, expand_to=x_1
        )
        d_sigma_t = expand_tensor_like(
            input_tensor=scheduler_output.d_sigma_t, expand_to=x_1
        )
        x_t = sigma_t * x_0 + alpha_t * x_1
        dx_t = d_sigma_t * x_0 + d_alpha_t * x_1
        return PathSample(x_t=x_t, dx_t=dx_t, x_1=x_1, x_0=x_0, t=t)
