# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# pyre-unsafe

try:
    import apex
except ImportError:
    print("apex is not installed")

# pyre-fixme[21]: Could not find module `mmcv.runner`.
from mmcv.runner import HOOKS, OptimizerHook


@HOOKS.register_module()
# pyre-fixme[11]: Annotation `OptimizerHook` is not defined as a type.
class DistOptimizerHook(OptimizerHook):
    """Optimizer hook for distributed training."""

    def __init__(
        self,
        update_interval: int = 1,
        grad_clip=None,
        coalesce: bool = True,
        bucket_size_mb=-1,
        use_fp16: bool = False,
    ) -> None:
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.update_interval = update_interval
        self.use_fp16 = use_fp16

    def before_run(self, runner) -> None:
        runner.optimizer.zero_grad()

    def after_train_iter(self, runner) -> None:
        runner.outputs["loss"] /= self.update_interval
        if self.use_fp16:
            # runner.outputs['loss'].backward()
            with apex.amp.scale_loss(
                runner.outputs["loss"], runner.optimizer
            ) as scaled_loss:
                scaled_loss.backward()
        else:
            runner.outputs["loss"].backward()
        # pyre-fixme[16]: `DistOptimizerHook` has no attribute `every_n_iters`.
        if self.every_n_iters(runner, self.update_interval):
            if self.grad_clip is not None:
                # pyre-fixme[16]: `DistOptimizerHook` has no attribute `clip_grads`.
                self.clip_grads(runner.model.parameters())
            runner.optimizer.step()
            runner.optimizer.zero_grad()
