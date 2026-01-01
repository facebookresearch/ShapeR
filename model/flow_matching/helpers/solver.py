# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.
#
# This file is derived from https://github.com/facebookresearch/flow_matching
# Licensed under CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/

# pyre-unsafe

import torch
from torch import Tensor
from torchdiffeq import odeint


class ODESolver:
    """A class to solve ordinary differential equations (ODEs) using a specified velocity model.

    This class utilizes a velocity field model to solve ODEs over a given time grid using numerical ode solvers.

    Args:
        velocity_model (Union[ModelWrapper, Callable]): a velocity field model receiving :math:`(x,t)` and returning :math:`u_t(x)`
    """

    def __init__(self, velocity_model):
        super().__init__()
        self.velocity_model = velocity_model

    def sample(
        self,
        x_init: Tensor,
        # pyre-fixme[9]: step_size has type `float`; used as `None`.
        step_size: float = None,
        method: str = "euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        enable_grad: bool = False,
        **model_extras,
    ):
        r"""Solve the ODE with the velocity field.

        Args:
            x_init (Tensor): initial conditions (e.g., source samples :math:`X_0 \sim p`). Shape: [batch_size, ...].
            step_size (Optional[float]): The step size. Must be None for adaptive step solvers.
            method (str): A method supported by torchdiffeq. Defaults to "euler". Other commonly used solvers are "dopri5", "midpoint" and "heun3". For a complete list, see torchdiffeq.
            atol (float): Absolute tolerance, used for adaptive step solvers.
            rtol (float): Relative tolerance, used for adaptive step solvers.
            time_grid (Tensor): The process is solved in the interval [min(time_grid, max(time_grid)] and if step_size is None then time discretization is set by the time grid. May specify a descending time_grid to solve in the reverse direction. Defaults to torch.tensor([0.0, 1.0]).
            return_intermediates (bool, optional): If True then return intermediate time steps according to time_grid. Defaults to False.
            enable_grad (bool, optional): Whether to compute gradients during sampling. Defaults to False.
            **model_extras: Additional input for the model.

        Returns:
          The last timestep when return_intermediates=False, otherwise all values specified in time_grid.
        """

        def ode_func(t, x):
            return self.velocity_model(x=x, t=t, **model_extras)

        ode_opts = {"step_size": step_size} if step_size is not None else {}

        with torch.set_grad_enabled(enable_grad):
            # Approximate ODE solution with numerical ODE solver
            sol = odeint(
                ode_func,
                x_init,
                time_grid,
                method=method,
                options=ode_opts,
                atol=atol,
                rtol=rtol,
            )

        if return_intermediates:
            return sol
        else:
            return sol[-1]
