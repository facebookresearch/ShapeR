# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.
#
# This file is derived from https://github.com/facebookresearch/flow_matching
# Licensed under CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/

# pyre-unsafe

from abc import ABC

from torch import nn, Tensor


class ModelWrapper(ABC, nn.Module):
    """
    This class is used to wrap around another model, adding custom forward pass logic.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        r"""
        This method defines how inputs should be passed through the wrapped model.
        Here, we're assuming that the wrapped model takes both :math:`x` and :math:`t` as input,
        along with any additional keyword arguments.

        Optional things to do here:
            - check that t is in the dimensions that the model is expecting.
            - add a custom forward pass logic.
            - call the wrapped model.

        | given x, t
        | returns the model output for input x at time t, with extra information `extra`.

        Args:
            x (Tensor): input data to the model (batch_size, ...).
            t (Tensor): time (batch_size).
            **extras: additional information forwarded to the model, e.g., text condition.

        Returns:
            Tensor: model output.
        """
        return self.model(x=x, t=t, **extras)
