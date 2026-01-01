# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import torch
import torchsparse
import torchsparse.nn as spnn
from einops import repeat

from torch import nn
from torch.nn import functional as F


def make_conv3d_sparse(
    channels_in,
    channels_out,
    kernel_size=3,
    num_groups=8,
    activation=spnn.ReLU,
):
    num_groups = min(num_groups, channels_out)
    block = nn.Sequential(
        spnn.Conv3d(channels_in, channels_out, kernel_size=kernel_size, stride=1),
        spnn.GroupNorm(num_groups, channels_out),
        activation(inplace=True),
    )
    return block


def make_conv3d_downscale_sparse(
    channels_in,
    channels_out,
    num_groups=8,
    activation=spnn.ReLU,
):
    num_groups = min(num_groups, channels_out)
    block = nn.Sequential(
        spnn.Conv3d(channels_in, channels_out, kernel_size=2, stride=2),
        spnn.GroupNorm(num_groups, channels_out),
        activation(inplace=True),
    )
    return block


class ResBlockSparse(nn.Module):
    def __init__(
        self,
        channels,
        num_groups=8,
        activation=spnn.ReLU,
    ):
        super().__init__()

        self.block0 = make_conv3d_sparse(
            channels, channels, num_groups=num_groups, activation=activation
        )
        self.block1 = make_conv3d_sparse(
            channels, channels, num_groups=num_groups, activation=activation
        )

    def forward(self, x):
        out = self.block0(x)
        out = self.block1(out)
        return x + out


def index_batched_sparse_tensor(sparse_tensor, index):
    """Index into a batched torchsparse.SparseTensor.

    Args:
        sparse_tensor: a torchsparse.SparseTensor that is the output of
            torchsparse.utils.collate.sparse_collate().
        index: int.

    Returns:
        torchsparse.SparseTensor with no batch dimension.
    """
    batch_mask = sparse_tensor.C[:, -1] == index
    coords = sparse_tensor.C[batch_mask, :-1]  # Get rid of batch dim
    feats = sparse_tensor.F[batch_mask]
    return torchsparse.SparseTensor(
        coords=coords,
        feats=feats,
        stride=sparse_tensor.s,
    )


def sparse_uncollate(sparse_tensor):
    """Un-Collate a batched torchsparse.SparseTensor.

    Args:
        sparse_tensor: a torchsparse.SparseTensor that is the output of
            torchsparse.utils.collate.sparse_collate().

    Returns:
        List[torchsparse.SparseTensor].
    """
    batch_size = sparse_tensor.C[:, -1].max() + 1

    sparse_tensor_list = []
    for b in range(batch_size):
        sparse_tensor_list.append(index_batched_sparse_tensor(sparse_tensor, b))
    return sparse_tensor_list


def vox_to_sequence(sparse_tensor):
    """Compute sequence from sparse point cloud.

    Args:
        sparse_tensor: torchsparse.SparseTensor.

    Returns:
        Dict with the following keys:
            seq: [B, maxlen, C] torch.FloatTensor.
            coords: [B, maxlen, 3] torch.IntTensor. To be used with embeddings for Transformers.
            mask: [B, maxlen] torch.BoolTensor. To be used with Transformers.
    """
    sparse_tensor_list = sparse_uncollate(sparse_tensor)
    batch_size = len(sparse_tensor_list)
    channels = sparse_tensor_list[0].F.shape[-1]
    maxlen = max([x.C.shape[0] for x in sparse_tensor_list])

    seq = torch.zeros(
        (batch_size, maxlen, channels),
        dtype=sparse_tensor.F.dtype,
        device=sparse_tensor.F.device,
    )
    coords = torch.zeros(
        (batch_size, maxlen, 3),
        dtype=sparse_tensor.C.dtype,
        device=sparse_tensor.C.device,
    )
    mask = torch.ones(
        (batch_size, maxlen),
        dtype=torch.bool,
        device=sparse_tensor.F.device,
    )
    for i in range(batch_size):
        sparse_tensor = sparse_tensor_list[i]

        # Get coords (divided by stride, so its in {0, 1, ...})
        coords_i = sparse_tensor.C  # [N_points_i, 3]
        coords_num = coords_i.shape[0]
        assert coords_num <= maxlen, f"coords_num: {coords_num} is too high..."

        # Get features
        feats = sparse_tensor.F  # [N_points_i, C]

        # Pad and save
        pad = maxlen - coords_num

        feats = F.pad(feats.T, (0, pad), value=0).T  # [maxlen, C]
        seq[i] = feats

        coords_i = F.pad(coords_i.T, (0, pad), value=0).T  # [maxlen, 3]
        coords[i] = coords_i

        mask[i, :coords_num] = False

    return {
        "seq": seq,
        "coords": coords,
        "mask": mask,
    }


def fourier_encode_vector(vec, num_bands=10, sample_rate=60):
    """Fourier encode a vector.

    Args:
        vec: [B, N, D] torch.FloatTensor.
        num_bands: int.
        sample_rate: int.

    Returns:
        [B, N, (2 * num_bands + 1) * D] torch.FloatTensor.
    """
    b, n, d = vec.shape
    samples = torch.linspace(1, sample_rate / 2, num_bands).to(vec.device) * torch.pi
    sines = torch.sin(samples[None, None, :, None] * vec[:, :, None, :])
    cosines = torch.cos(samples[None, None, :, None] * vec[:, :, None, :])

    encoding = torch.stack([sines, cosines], dim=3).reshape(b, n, 2 * num_bands, d)
    encoding = torch.cat([vec[:, :, None, :], encoding], dim=2)
    return encoding.flatten(2)


class ResNet3DSparse(nn.Module):
    def __init__(self, dim_in, dim_out, layers):
        super().__init__()

        self.stem = nn.Sequential(
            make_conv3d_sparse(dim_in, layers[0], kernel_size=7),
            ResBlockSparse(layers[0]),
        )

        # Number of down-convs is len(layers) - 1
        blocks = []
        for i in range(len(layers) - 1):
            blocks.append(
                nn.Sequential(
                    make_conv3d_downscale_sparse(layers[i], layers[i + 1]),
                    ResBlockSparse(layers[i + 1]),
                    ResBlockSparse(layers[i + 1]),
                )
            )
        self.blocks = nn.Sequential(*blocks)

        self.bottleneck = nn.Sequential(
            nn.Linear(layers[-1], 2 * layers[-1]),
            nn.GroupNorm(8, 2 * layers[-1]),  # 8 is default num_groups for GroupNorm
            nn.ReLU(inplace=True),
            nn.Linear(2 * layers[-1], dim_out),
        )

    def forward(self, x):
        out = self.stem(x)
        out = self.blocks(out)
        out.F = self.bottleneck(out.F)  # bottleneck applied to features only
        return out


class PointCloudEncoder(nn.Module):
    def __init__(
        self,
        input_channels,
        d_model,
        conv_layers,
        num_bins,
    ):
        """Point Cloud Encoder.

        Args:
            input_channels: int.
            d_model: int.
            conv_layers: List[int].
            num_bins: int.
        """

        super().__init__()

        self.sparse_resnet = ResNet3DSparse(
            dim_in=input_channels,
            dim_out=d_model,
            layers=conv_layers,
        )
        downconvs = len(conv_layers) - 1
        res_reduction = 2**downconvs  # voxel resolution reduction
        self.reduced_grid_size = int(num_bins / res_reduction)
        self.input_proj = nn.Linear(d_model + 63, d_model)  # 63 for fourier encoding

        self.pad_embedding = nn.Parameter(
            torch.empty(d_model).normal_(std=1e-3), requires_grad=True
        )
        # the following is a legacy parameter
        self.extra_embedding = nn.Parameter(torch.empty(d_model).normal_(std=0.02))

    def forward(self, point_cloud: torchsparse.SparseTensor):
        """Forward function.

        Args:
            point_cloud: torchsparse.SparseTensor.

        Returns: a Dict with the following keys:
            context: [B, maxlen, d_model] torch.FloatTensor.
            context_mask: [B, maxlen] torch.BoolTensor. True means ignore.
        """
        outputs = self.sparse_resnet(point_cloud)
        outputs = vox_to_sequence(outputs)

        context = outputs["seq"]
        context_mask = outputs["mask"]

        # set padding
        context[context_mask, :] = self.pad_embedding.to(context.dtype)

        coords = outputs["coords"]
        coords_normalised = coords / (self.reduced_grid_size - 1)
        encoded_coords = fourier_encode_vector(coords_normalised)

        context = torch.cat([context, encoded_coords], dim=-1)
        context = self.input_proj(context)

        # legacy parameter
        context = context + repeat(self.extra_embedding.to(context.dtype), "d -> 1 1 d")

        return {
            "context": context,
            "context_mask": context_mask,
        }
