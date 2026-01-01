# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import math
from typing import List

import torch
from torch import nn


class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: int,
        n_res_blocks: int,
        channel_multipliers: List[int],
    ):
        """
        :param in_channels: is the number of channels in the input feature map
        :param out_channels: is the number of channels in the output feature map
        :param channels: is the base channel count for the model
        :param n_res_blocks: number of residual blocks at each level
        :param channel_multipliers: are the multiplicative factors for number of channels for each level
        """
        super().__init__()
        self.channels = channels

        # Number of levels
        levels = len(channel_multipliers)

        # Input half of the U-Net
        self.input_blocks = nn.ModuleList()
        self.input_blocks.append(nn.Conv2d(in_channels, channels, 3, padding=1))

        # Number of channels at each block in the input half of U-Net
        input_block_channels = [channels]
        # Number of channels at each level
        channels_list = [channels * m for m in channel_multipliers]
        # Prepare levels
        for i in range(levels):
            # Add the residual blocks and attentions
            for _ in range(n_res_blocks):
                # Residual block maps from previous number of channels to the number of
                # channels in the current level
                layer = ResBlock(channels, out_channels=channels_list[i])
                channels = channels_list[i]
                self.input_blocks.append(layer)
                input_block_channels.append(channels)
            # Down sample at all levels except last
            if i != levels - 1:
                self.input_blocks.append(DownSample(channels))
                input_block_channels.append(channels)

        self.middle_block = ResBlock(channels)

        # Second half of the U-Net
        self.output_blocks = nn.ModuleList([])
        # Prepare levels in reverse order
        for i in reversed(range(levels)):
            # Add the residual blocks and attentions
            for j in range(n_res_blocks + 1):
                # Residual block maps from previous number of channels plus the
                # skip connections from the input half of U-Net to the number of
                # channels in the current level.
                layers = [
                    ResBlock(
                        channels + input_block_channels.pop(),
                        out_channels=channels_list[i],
                    )
                ]
                channels = channels_list[i]
                # Up-sample at every level after last residual block
                # except the last one.
                # Note that we are iterating in reverse; i.e. `i == 0` is the last.
                if i != 0 and j == n_res_blocks:
                    # pyre-fixme[6]: For 1st argument expected `ResBlock` but got
                    #  `UpSample`.
                    layers.append(UpSample(channels))
                # Add to the output half of the U-Net
                self.output_blocks.append(nn.Sequential(*layers))

        # Final normalization and $3 \times 3$ convolution
        self.out = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map of shape `[batch_size, channels, width, height]`
        :param time_steps: are the time steps of shape `[batch_size]`
        :param cond: conditioning of shape `[batch_size, n_cond, d_cond]`
        """
        # To store the input half outputs for skip connections
        x_input_block = []

        # Input half of the U-Net
        for module in self.input_blocks:
            x = module(x)
            x_input_block.append(x)
        # Middle of the U-Net
        x = self.middle_block(x)
        # Output half of the U-Net
        for module in self.output_blocks:
            x = torch.cat([x, x_input_block.pop()], dim=1)
            x = module(x)

        # Final normalization and $3 \times 3$ convolution
        return self.out(x)


class UpSample(nn.Module):
    """
    ### Up-sampling layer
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # $3 \times 3$ convolution mapping
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Up-sample by a factor of $2$
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        # Apply convolution
        return self.conv(x)


class DownSample(nn.Module):
    """
    ## Down-sampling layer
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # $3 \times 3$ convolution with stride length of $2$ to down-sample by a factor of $2$
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Apply convolution
        return self.op(x)


class ResBlock(nn.Module):
    """
    ## ResNet Block
    """

    def __init__(self, channels: int, *, out_channels=None, num_norm_groups=32):
        """
        :param channels: the number of input channels
        :param d_t_emb: the size of timestep embeddings
        :param out_channels: is the number of out channels. defaults to `channels.
        """
        super().__init__()
        # `out_channels` not specified
        if out_channels is None:
            out_channels = channels

        # First normalization and convolution
        self.in_layers = nn.Sequential(
            normalization(channels, num_norm_groups),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )

        # Final convolution layer
        self.out_layers = nn.Sequential(
            normalization(out_channels, num_norm_groups),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        # `channels` to `out_channels` mapping layer for residual connection
        if out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, out_channels, 1)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        :param t_emb: is the time step embeddings of shape `[batch_size, d_t_emb]`
        """
        # Initial convolution
        h = self.in_layers(x)
        # Final convolution
        h = self.out_layers(h)
        # Add skip connection
        return self.skip_connection(x) + h


class GroupNorm32(nn.GroupNorm):
    """
    ### Group normalization with float32 casting
    """

    def forward(self, x):
        return super().forward(x)


def normalization(channels, num_groups=32):
    """
    ### Group normalization

    This is a helper function, with fixed number of groups..
    """
    return GroupNorm32(num_groups, channels)


class MixerUNetModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dino_in_channels: int,
        out_channels: int,
        channels: int,
        n_res_blocks: int,
        channel_multipliers: List[int],
        end_at: int = 0,
    ):
        """
        :param in_channels: is the number of channels in the input feature map
        :param out_channels: is the number of channels in the output feature map
        :param channels: is the base channel count for the model
        :param n_res_blocks: number of residual blocks at each level
        :param channel_multipliers: are the multiplicative factors for number of channels for each level
        """
        super().__init__()
        self.channels = channels

        # Number of levels
        levels = len(channel_multipliers)

        # Input half of the U-Net
        self.input_blocks = nn.ModuleList()
        self.input_blocks.append(nn.Conv2d(in_channels, channels, 3, padding=1))

        # Number of channels at each block in the input half of U-Net
        input_block_channels = [channels]
        # Number of channels at each level
        channels_list = [channels * m for m in channel_multipliers]
        # Prepare levels
        for i in range(levels):
            # Add the residual blocks and attentions
            for _ in range(n_res_blocks):
                # Residual block maps from previous number of channels to the number of
                # channels in the current level
                layer = ResBlock(
                    channels, out_channels=channels_list[i], num_norm_groups=16
                )
                channels = channels_list[i]
                self.input_blocks.append(layer)
                input_block_channels.append(channels)
            # Down sample at all levels except last
            if i != levels - 1:
                self.input_blocks.append(DownSample(channels))
                input_block_channels.append(channels)

        self.middle_block = ResBlock(
            channels * 2, out_channels=channels, num_norm_groups=16
        )
        self.middle_projection_layer = nn.Conv2d(
            dino_in_channels, channels, kernel_size=1, stride=1, padding=0
        )

        # Second half of the U-Net
        self.output_blocks = nn.ModuleList([])
        # Prepare levels in reverse order
        for i in reversed(range(end_at, levels)):
            # Add the residual blocks and attentions
            for j in range(n_res_blocks + 1):
                # Residual block maps from previous number of channels plus the
                # skip connections from the input half of U-Net to the number of
                # channels in the current level.
                layers = [
                    ResBlock(
                        channels + input_block_channels.pop(),
                        out_channels=channels_list[i],
                        num_norm_groups=16,
                    )
                ]
                channels = channels_list[i]
                # Up-sample at every level after last residual block
                # except the last one.
                # Note that we are iterating in reverse; i.e. `i == 0` is the last.
                if i != end_at and j == n_res_blocks:
                    # pyre-fixme[6]: For 1st argument expected `ResBlock` but got
                    #  `UpSample`.
                    layers.append(UpSample(channels))
                # Add to the output half of the U-Net
                self.output_blocks.append(nn.Sequential(*layers))
        # Final normalization and $3 \times 3$ convolution
        self.out = nn.Sequential(
            normalization(channels, num_groups=16),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, dino_image_features: torch.Tensor):
        """
        :param x: is the input feature map of shape `[batch_size, channels, width, height]`
        :param time_steps: are the time steps of shape `[batch_size]`
        :param cond: conditioning of shape `[batch_size, n_cond, d_cond]`
        """
        # To store the input half outputs for skip connections
        x_input_block = []

        # Input half of the U-Net
        for module in self.input_blocks:
            x = module(x)
            x_input_block.append(x)

        # Middle of the U-Net
        if (
            dino_image_features.shape[-1] != x.shape[-1]
            or dino_image_features.shape[-2] != x.shape[-2]
        ):
            dino_image_features = torch.nn.functional.interpolate(
                dino_image_features, size=x.shape[-2:], mode="nearest"
            )
        projected_dino = self.middle_projection_layer(dino_image_features)

        x = self.middle_block(torch.cat([x, projected_dino], dim=1))
        # Output half of the U-Net
        for module in self.output_blocks:
            x = torch.cat([x, x_input_block.pop()], dim=1)
            x = module(x)

        # Final normalization and $3 \times 3$ convolution
        return self.out(x)


class MaskDownsamplingNet(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        flatten_embedding=True,
    ):
        super().__init__()
        if type(patch_size) is int:
            patch_size = (patch_size, patch_size)
        if type(img_size) is int:
            img_size = (img_size, img_size)
        self.target_image_resolution = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.num_levels = int(math.ceil(math.log2(patch_size[0])))
        base_channels = 16
        channels_list = [int(base_channels * 2**i) for i in range(self.num_levels)]
        self.input_blocks = nn.ModuleList()
        self.proj_in = nn.Conv2d(
            in_chans, base_channels, kernel_size=3, padding=1, stride=1
        )
        channels = base_channels
        for i in range(self.num_levels):
            layer = ResBlock(
                channels, out_channels=channels_list[i], num_norm_groups=base_channels
            )
            channels = channels_list[i]
            self.input_blocks.append(layer)
            self.input_blocks.append(DownSample(channels))

        # last level

        self.input_blocks.append(
            nn.Sequential(
                normalization(channels, num_groups=base_channels),
                nn.SiLU(),
                nn.Conv2d(channels, embed_dim, 1, padding=0, stride=1),
            )
        )
        self.flatten = flatten_embedding
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.proj_in(x)
        for module in self.input_blocks:
            x = module(x)
        x = torch.nn.functional.interpolate(
            x, size=self.target_image_resolution, mode="nearest"
        )
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        return x


class ConvResize(nn.Module):
    def __init__(self, in_chans, target_size, feature_dim, flatten=True):
        super().__init__()
        self.target_size = (target_size, target_size)
        self.proj_in = nn.Conv2d(
            in_chans, feature_dim, kernel_size=3, padding=1, stride=1
        )
        self.apply(initialize_weights)
        self.flatten = flatten

    def forward(self, x):
        x = self.proj_in(x)
        x = torch.nn.functional.interpolate(x, size=self.target_size)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        return x


def initialize_weights(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
