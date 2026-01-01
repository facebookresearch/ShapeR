# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# pyre-unsafe

from io import BytesIO
from typing import Any

from PIL import Image


class Decoder:
    def decode(self) -> Any:
        raise NotImplementedError


class ImageDataDecoder(Decoder):
    def __init__(self, image_data: bytes) -> None:
        self._image_data = image_data

    # pyre-fixme[11]: Annotation `Image` is not defined as a type.
    def decode(self) -> Image:
        f = BytesIO(self._image_data)
        return Image.open(f).convert(mode="RGB")


class TargetDecoder(Decoder):
    def __init__(self, target: Any) -> None:
        self._target = target

    def decode(self) -> Any:
        return self._target
