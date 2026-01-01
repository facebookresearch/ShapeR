# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import torch


class PointCloud:
    def __init__(self, points):
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

    def translate(self, translation_vector):
        """Translate point cloud.

        Args:
            translation_vector: [3] torch.FloatTensor of XYZ translation vector.
        """
        points_xyz = self.points[:, :3]
        points_rest = self.points[:, 3:]
        translated_points = points_xyz + translation_vector
        self.points = torch.cat([translated_points, points_rest], dim=1)

    def normalize_and_discretize(self, num_bins, object_bounds=None):
        """Normalize and Discretize the point cloud.

        Args:
            num_bins: int.
            normalization_extent:
        """
        original_points = self.points.clone()

        if object_bounds is None:
            # assuming -1 to 1 extent
            if isinstance(self.points, torch.Tensor):
                object_bounds = torch.FloatTensor(
                    [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]
                ).to(self.points)
            else:
                object_bounds = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])

        normalization_extent = object_bounds[1] - object_bounds[0]

        # Normalize and translate to positive quadrant
        normalized_points = (original_points - object_bounds[0]) / normalization_extent

        # Discretise
        voxel_coords = (normalized_points * num_bins).round().long()
        voxel_coords = voxel_coords.clamp(max=num_bins - 1)

        # Get unique voxel coordinates
        unique_voxel_coords, inverse, unique_voxel_counts = torch.unique(
            voxel_coords, dim=0, return_inverse=True, return_counts=True
        )

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
        self.coords = unique_voxel_coords  # in {0, 1, ..., num_bins - 1}
