# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# pyre-unsafe

# pyre-fixme[21]: Could not find module `mmcv.runner`.
from mmcv.runner import BaseModule

# pyre-fixme[21]: Could not find module `mmseg.models.builder`.
from mmseg.models.builder import BACKBONES


@BACKBONES.register_module()
# pyre-fixme[11]: Annotation `BaseModule` is not defined as a type.
class DinoVisionTransformer(BaseModule):
    """Vision Transformer."""

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
