# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch


def _basic_init(module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)


def _zero_init(module):
    if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv3d)):
        torch.nn.init.constant_(module.weight, 0)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)


class Conv2Or3dChannelLast(torch.nn.Module):
    """
    nn.Upsample a tensor with channel last format.
    """

    def __init__(self, dims: int, *args, **kwargs):
        super().__init__()
        if dims == 2:
            self._conv = torch.nn.Conv2d(*args, **kwargs)
        elif dims == 3:
            self._conv = torch.nn.Conv3d(*args, **kwargs)
        else:
            raise ValueError("Unsupported number of dimensions")

    def forward(self, x: torch.Tensor):
        if x.ndim == 4:
            x_permuted = x.permute(0, 3, 1, 2)
        elif x.ndim == 5:
            x_permuted = x.permute(0, 4, 1, 2, 3)
        else:
            raise ValueError("Unsupported input shape")

        y = self._conv(x_permuted)

        if y.ndim == 4:
            return y.permute(0, 2, 3, 1)
        elif y.ndim == 5:
            return y.permute(0, 2, 3, 4, 1)


class UpsampleChannelLast(torch.nn.Module):
    """
    nn.Upsample a tensor with channel last format.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._upsample = torch.nn.Upsample(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        if x.ndim == 4:
            x_permuted = x.permute(0, 3, 1, 2)
        elif x.ndim == 5:
            x_permuted = x.permute(0, 4, 1, 2, 3)
        else:
            raise ValueError("Unsupported input shape")

        x_upsampled = self._upsample(x_permuted)

        if x.ndim == 4:
            return x_upsampled.permute(0, 2, 3, 1)
        elif x.ndim == 5:
            return x_upsampled.permute(0, 2, 3, 4, 1)


class UpsampleX2Conv2dResBlock(torch.nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self._upsample = UpsampleChannelLast(scale_factor=2, mode="nearest")
        self._conv_0 = Conv2Or3dChannelLast(
            2, n_feat, n_feat, kernel_size=3, stride=1, padding=1
        )
        self._conv_1 = Conv2Or3dChannelLast(
            2, n_feat, n_feat, kernel_size=3, stride=1, padding=1
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x_upsampled = self._upsample(x)
        x_conv_0 = self._conv_0(x_upsampled)
        x_conv_1 = self.relu(self._conv_1(x_conv_0))
        return x_conv_1


class UpsampleX4Conv2dResBlock(torch.nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self._upsample_0 = UpsampleChannelLast(scale_factor=2, mode="nearest")
        self._upsample_1 = UpsampleChannelLast(scale_factor=2, mode="nearest")
        self._conv_0 = Conv2Or3dChannelLast(
            2, n_feat, n_feat, kernel_size=3, stride=1, padding=1
        )
        self._conv_1 = Conv2Or3dChannelLast(
            2, n_feat, n_feat, kernel_size=3, stride=1, padding=1
        )
        self._conv_2 = Conv2Or3dChannelLast(
            2, n_feat, n_feat, kernel_size=3, stride=1, padding=1
        )
        self._conv_3 = Conv2Or3dChannelLast(
            2, n_feat, n_feat, kernel_size=3, stride=1, padding=1
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x_upsampled = self._upsample_0(x)
        x_conv_0 = self._conv_0(x_upsampled)
        x_conv_1 = self.relu(self._conv_1(x_conv_0))
        x_upsampled = self._upsample_1(x_conv_1)
        x_conv_2 = self._conv_2(x_upsampled)
        x_conv_3 = self.relu(self._conv_3(x_conv_2))
        return x_conv_3


class UpsampleX8Conv2dResBlock(torch.nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self._upsample_0 = UpsampleChannelLast(scale_factor=2, mode="nearest")
        self._upsample_1 = UpsampleChannelLast(scale_factor=2, mode="nearest")
        self._upsample_2 = UpsampleChannelLast(scale_factor=2, mode="nearest")
        self._conv_0 = Conv2Or3dChannelLast(
            2, n_feat, n_feat, kernel_size=3, stride=1, padding=1
        )
        self._conv_1 = Conv2Or3dChannelLast(
            2, n_feat, n_feat, kernel_size=3, stride=1, padding=1
        )
        self._conv_2 = Conv2Or3dChannelLast(
            2, n_feat, n_feat, kernel_size=3, stride=1, padding=1
        )
        self._conv_3 = Conv2Or3dChannelLast(
            2, n_feat, n_feat, kernel_size=3, stride=1, padding=1
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x_upsampled = self._upsample_0(x)
        x_conv_0 = self._conv_0(x_upsampled)
        x_conv_1 = self.relu(self._conv_1(x_conv_0))
        x_upsampled = self._upsample_1(x_conv_1)
        x_upsampled = self._upsample_2(x_upsampled)
        x_conv_2 = self._conv_2(x_upsampled)
        x_conv_3 = self.relu(self._conv_3(x_conv_2))
        return x_conv_3
