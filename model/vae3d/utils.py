# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.
#
# This file is inspired by the DORA-VAE project.
# Original source: https://github.com/Seed3D/Dora
# Original license: Apache License 2.0

# pyre-unsafe

from typing import List, Tuple, Union

import numpy as np

import torch
import trimesh
from einops import repeat
from skimage import measure
from torch import nn
from tqdm import tqdm


def generate_dense_grid_points(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    octree_depth: int,
    indexing: str = "ij",
):
    length = bbox_max - bbox_min
    num_cells = np.exp2(octree_depth)
    x = np.linspace(bbox_min[0], bbox_max[0], int(num_cells) + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], int(num_cells) + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], int(num_cells) + 1, dtype=np.float32)
    # pyre-fixme[6]: For 4th argument expected `Union[Literal['ij'], Literal['xy']]`
    #  but got `str`.
    [xs, ys, zs] = np.meshgrid(x, y, z, indexing=indexing)
    xyz = np.stack((xs, ys, zs), axis=-1)
    xyz = xyz.reshape(-1, 3)
    grid_size = [
        int(num_cells) + 1,
        int(num_cells) + 1,
        int(num_cells) + 1,
    ]
    return xyz, grid_size, length, xs, ys, zs


class DiagonalGaussianDistribution:
    def __init__(
        self,
        parameters,
        deterministic=False,
        feat_dim=1,
    ):
        self.feat_dim = feat_dim
        self.parameters = parameters

        if isinstance(parameters, list):
            self.mean = parameters[0]
            self.logvar = parameters[1]
        else:
            self.mean, self.logvar = torch.chunk(parameters, 2, dim=feat_dim)

        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self):
        x = self.mean + self.std * torch.randn_like(self.mean)
        return x

    def kl(self, other=None, dims=(1, 2)):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.mean(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=dims
                )
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=dims,
                )

    def nll(self, sample, dims=(1, 2)):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean


class FourierEmbedder(nn.Module):
    def __init__(
        self,
        num_freqs: int = 6,
        logspace: bool = True,
        input_dim: int = 3,
        include_input: bool = True,
        include_pi: bool = True,
    ) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        else:
            frequencies = torch.linspace(
                1.0, 2.0 ** (num_freqs - 1), num_freqs, dtype=torch.float32
            )

        if include_pi:
            frequencies *= torch.pi

        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs

        self.out_dim = self.get_dims(input_dim)

    def get_dims(self, input_dim):
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        out_dim = input_dim * (self.num_freqs * 2 + temp)

        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_freqs > 0:
            embed = (x[..., None].contiguous() * self.frequencies).view(
                *x.shape[:-1], -1
            )
            if self.include_input:
                return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            else:
                return torch.cat((embed.sin(), embed.cos()), dim=-1)
        else:
            return x


def get_embedder(embed_type="fourier", num_freqs=-1, input_dim=3, include_pi=True):
    if embed_type == "identity" or (embed_type == "fourier" and num_freqs == -1):
        return nn.Identity(), input_dim

    elif embed_type == "fourier":
        embedder_obj = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)

    else:
        raise ValueError(f"{embed_type} is not valid. Currently only supports fourier")
    return embedder_obj


class AutoEncoder(nn.Module):
    def __init__(self, embed_dim, use_udf_extraction):
        super().__init__()
        self.embed_dim = embed_dim
        self.pre_kl = None
        self.use_udf_extraction = use_udf_extraction
        self.udf_iso = 0.4

    def encode(self, x, is_training):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError

    def query(self, queries, latents):
        raise NotImplementedError

    def infer_mesh(self, batch):
        raise NotImplementedError

    @torch.no_grad()
    def extract_mesh(
        self,
        latents,
        bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        octree_depth=8,
        num_chunks=32768,
    ):
        bbox_min = np.array(bounds[0:3])
        bbox_max = np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min
        xyz_samples, grid_size, length, xs, ys, zs = generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_depth=octree_depth,
            indexing="ij",
        )
        xyz_samples = torch.FloatTensor(xyz_samples)
        batch_size = latents.shape[0]

        batch_sdf = []

        for start in tqdm(
            range(0, xyz_samples.shape[0], num_chunks), desc="mesh_extract"
        ):
            queries = xyz_samples[start : start + num_chunks, :].to(
                dtype=latents.dtype, device=latents.device
            )
            batch_queries = repeat(queries, "p c -> b p c", b=batch_size)

            sdf = self.query(batch_queries, latents)
            batch_sdf.append(sdf.cpu())

        grid_sdf = (
            torch.cat(batch_sdf, dim=1)
            .view((batch_size, grid_size[0], grid_size[1], grid_size[2]))
            .float()
            .numpy()
        )

        mesh = []
        for i in range(batch_size):
            try:
                if self.use_udf_extraction:
                    vertices, faces, normals, _ = measure.marching_cubes(
                        np.abs(grid_sdf[i]),
                        self.udf_iso,
                        method="lewiner",
                        gradient_direction="ascent",
                    )
                else:
                    vertices, faces, normals, _ = measure.marching_cubes(
                        grid_sdf[i],
                        0,
                        method="lewiner",
                        gradient_direction="ascent",
                    )
                vertices = vertices / grid_size * bbox_size + bbox_min
                faces = faces[:, [2, 1, 0]]
                mesh.append(
                    trimesh.Trimesh(
                        vertices=vertices.astype(np.float32),
                        faces=np.ascontiguousarray(faces),
                    )
                )
            except Exception:
                mesh.append(trimesh.creation.icosphere(radius=0.01, subdivisions=2))
        return mesh
