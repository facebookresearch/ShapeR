# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Point cloud processing utilities.

Handles discretization, normalization, and conversion to SparseTensor for
efficient 3D sparse convolutions via torchsparse.
"""

# pyre-unsafe

import numpy as np

import torch
import torchsparse
from torchsparse.utils.collate import sparse_collate


class PointCloud:
    def __init__(self, points) -> None:
        """A class that wraps some point cloud functionality.

        Args:
            points: [N, 3] torch.FloatTensor of XYZ coordinates of the point cloud.
        """
        self.points = points

    def extent(self):
        """Compute extent of point cloud.

        Returns:
            Dict with the following keys: {min/max/size}_{x/y/z}.
                Values are floats.
        """

        min_x = 1e6
        min_y = 1e6
        min_z = 1e6
        max_x = -1e6
        max_y = -1e6
        max_z = -1e6

        points_min = self.points.min(dim=0)[0]
        min_x = min(min_x, points_min[0])
        min_y = min(min_y, points_min[1])
        min_z = min(min_z, points_min[2])

        points_max = self.points.max(dim=0)[0]
        max_x = max(max_x, points_max[0])
        max_y = max(max_y, points_max[1])
        max_z = max(max_z, points_max[2])

        return {
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "min_z": min_z,
            "max_z": max_z,
            "size_x": max(max_x - min_x, 0),
            "size_y": max(max_y - min_y, 0),
            "size_z": max(max_z - min_z, 0),
        }

    def translate(self, translation_vector) -> None:
        """Translate point cloud.

        Args:
            translation_vector: [3] torch.FloatTensor of XYZ translation vector.
        """
        points_xyz = self.points[:, :3]
        points_rest = self.points[:, 3:]
        translated_points = points_xyz + translation_vector
        self.points = torch.cat([translated_points, points_rest], dim=1)

    def normalize_and_discretize(self, num_bins, object_bounds=None) -> None:
        """Normalize and Discretize the point cloud.

        Args:
            num_bins: int.
            normalization_extent:
        """
        original_points = self.points.clone()

        if object_bounds is None:
            # assuming -1 to 1 extent
            object_bounds = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])

        normalization_extent = object_bounds[1] - object_bounds[0]

        # Normalize and translate to positive quadrant
        normalized_points = (original_points - object_bounds[0]) / normalization_extent

        # Discretise
        voxel_coords = (normalized_points * num_bins).round().long()
        voxel_coords = voxel_coords.clamp(max=num_bins - 1)

        # Get unique voxel coordinates
        unique_voxel_coords, inverse, unique_voxel_counts = np.unique(
            voxel_coords.numpy(), axis=0, return_inverse=True, return_counts=True
        )
        unique_voxel_coords = torch.as_tensor(unique_voxel_coords)
        inverse = torch.as_tensor(inverse)
        unique_voxel_counts = torch.as_tensor(unique_voxel_counts)

        # Average of points falling in the same bin
        discretised_original_points = torch.stack(
            [
                torch.bincount(inverse, weights=original_points[:, i])
                / unique_voxel_counts
                for i in range(original_points.shape[1])
            ],
            dim=1,
        )

        self.points = discretised_original_points
        # pyre-fixme[16]: `PointCloud` has no attribute `coords`.
        self.coords = unique_voxel_coords  # in {0, 1, ..., num_bins - 1}


def preprocess_point_cloud(
    point_cloud_batch,
    num_bins,
    object_bboxes=None,
    push_to_positive_quadrant: bool = False,
):
    """Preprocess the point cloud to be fed into the encoder.

    Args:
        point_cloud: [B, N, 3] torch.FloatTensor.

    Returns:
        sparse_tensor: torchsparse.SparseTensor.
    """
    pc_sparse_tensors = []

    for b in range(len(point_cloud_batch)):
        point_cloud = PointCloud(point_cloud_batch[b])

        if push_to_positive_quadrant:
            # Push to positive quadrant
            extent = point_cloud.extent()
            pc_min = [extent["min_x"], extent["min_y"], extent["min_z"]]
            pc_min = torch.as_tensor(pc_min)
            point_cloud.translate(-pc_min)

        # Normalize / Discretize it
        point_cloud.normalize_and_discretize(
            num_bins, object_bboxes[b] if object_bboxes is not None else None
        )

        # Convert to torchsparse.SparseTensor
        pc_sparse_tensor = torchsparse.SparseTensor(
            # pyre-fixme[16]: `PointCloud` has no attribute `coords`.
            coords=point_cloud.coords.int(),
            feats=point_cloud.points.float(),
        )

        pc_sparse_tensors.append(pc_sparse_tensor)

    pc_sparse_tensor_batch = sparse_collate(pc_sparse_tensors)

    return pc_sparse_tensor_batch
