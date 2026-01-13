# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Dataset classes for loading and preprocessing ShapeR samples."""

import os
import pickle

import numpy as np
import torch
import torchsparse
from dataset.image_processor import (
    crop_pad_preselected_views_with_background,
    get_image_data_based_on_strategy,
)
from dataset.point_cloud import preprocess_point_cloud


class InferenceDataset(torch.utils.data.Dataset):
    """
    Dataset for loading preprocessed pickle samples for ShapeR inference.

    Each sample contains: point cloud, images, camera params, and optionally GT mesh.
    Supports view selection strategies: 'cluster', 'last_n', 'view_angle'.
    """

    def __init__(self, config, paths, override_num_views=None) -> None:
        super().__init__()
        self.config = config
        if override_num_views is not None:
            self.config.dataset.num_views = override_num_views
        self.paths = paths
        self.length = len(self.paths)
        self.strategy = "cluster"

    def set_strategy(self, strategy) -> None:
        self.strategy = strategy

    def __len__(self) -> int:
        return self.length

    def get_caption(self, pkl_sample):
        if "caption" in pkl_sample:
            return pkl_sample["caption"]
        elif "category" in pkl_sample:
            return f"{pkl_sample['category'].lower()}"
        return "a 3D object"

    def __getitem__(self, idx):
        sample_path = self.paths[idx]
        pkl_sample = pickle.load(open(sample_path, "rb"))

        semi_dense_points = pkl_sample["points_model"].numpy()
        bounds = pkl_sample["bounds"].numpy()
        semi_dense_theta = pkl_sample["inv_dist_std"].numpy()[:]
        semi_dense_phi = pkl_sample["dist_std"].numpy()[:]
        scale = 0.9 / np.max(bounds)
        semi_dense_points[:, :3] *= scale

        valid = np.all(np.abs(semi_dense_points) <= 1.0, axis=-1)
        semi_dense_points = semi_dense_points[valid, :]
        semi_dense_theta = semi_dense_theta[valid]
        semi_dense_phi = semi_dense_phi[valid]

        # threshold the semi-dense points
        selected_semi_dense_points = semi_dense_points

        thres_theta = self.config.dataset.semi_dense_threshold_theta
        thres_phi = self.config.dataset.semi_dense_threshold_phi

        filter_semi_dense = np.logical_and(
            semi_dense_theta <= thres_theta,
            semi_dense_phi <= thres_phi,
        )

        if filter_semi_dense.shape[0] >= 3:  # atleast 3 points
            selected_semi_dense_points = semi_dense_points[filter_semi_dense]
        else:
            selected_semi_dense_points = semi_dense_points

        sample = {
            "index": idx,
            "name": os.path.basename(self.paths[idx]).split(".")[0],
            "semi_dense_points": selected_semi_dense_points,
            "caption": self.get_caption(pkl_sample),
            "scale": scale,
        }

        # check if images are present
        if "image_data" in pkl_sample and self.config.dataset.load_image_mode != "none":
            selected_view_imgs_list = []
            rectified_masks_list = []
            selected_view_ext_list = []
            selected_view_int_list = []
            selected_view_uncropped_pil_bimgs_list = []
            (
                rectified_images,
                rectified_point_masks,
                rectified_camera_params,
                selected_view_ext,
            ) = get_image_data_based_on_strategy(
                pkl_sample,
                self.config.dataset.num_views,
                scale,
                is_rgb=False,
                strategy=self.strategy,
            )
            (
                selected_view_imgs,
                selected_view_uncropped_pil_bimgs,
                rectified_masks,
                selected_view_int,
            ) = crop_pad_preselected_views_with_background(
                rectified_images,
                rectified_point_masks,
                rectified_camera_params,
                self.config.encoder.dino_image_size,
                add_point_locations=self.config.dataset.load_image_mode
                == "composite_points",
            )
            selected_view_imgs_list.append(selected_view_imgs)
            rectified_masks_list.append(rectified_masks)
            selected_view_ext_list.append(selected_view_ext)
            selected_view_int_list.append(selected_view_int)
            selected_view_uncropped_pil_bimgs_list.extend(
                selected_view_uncropped_pil_bimgs
            )

            sample["images"] = np.concatenate(selected_view_imgs_list, axis=0)
            sample["masks_ingest"] = np.concatenate(rectified_masks_list, axis=0)
            sample["boxes_ingest"] = get_boxes_from_masks(sample["masks_ingest"])
            sample["camera_extrinsics"] = np.concatenate(selected_view_ext_list, axis=0)
            sample["camera_intrinsics"] = np.concatenate(selected_view_int_list, axis=0)

        # check if gt mesh is present
        if "mesh_vertices" in pkl_sample:
            sample["vertices"] = pkl_sample["mesh_vertices"].numpy() * scale
            sample["faces"] = pkl_sample["mesh_faces"].numpy()

        return sample

    def rescale_back(self, idx, mesh, do_transform_to_world: bool = False):
        """
        Transform predicted mesh from normalized coords back to original scale.

        The model predicts in [-0.9, 0.9] normalized space. This undoes that
        normalization and optionally transforms to world coordinates.
        """
        sample_path = self.paths[idx]
        pkl_sample = pickle.load(open(sample_path, "rb"))
        bounds = pkl_sample["bounds"].numpy()
        scale = 0.9 / np.max(bounds)
        mesh.apply_scale(1 / scale)

        if do_transform_to_world:
            if "T_model_world" in pkl_sample:
                T_model_world = pkl_sample["T_model_world"]
                mesh.apply_transform(T_model_world.inverse().numpy())
            elif "T_zup_obj" in pkl_sample:
                T_zup_obj = pkl_sample["T_zup_obj"]
                mesh.apply_transform(T_zup_obj.inverse().numpy())

        return mesh

    def custom_collate(self, batch):
        """
        Custom collate function for DataLoader.

        Handles: point clouds -> SparseTensor, vertices/faces -> lists (variable size).
        """
        batch_sdp = preprocess_point_cloud(
            [torch.tensor(b["semi_dense_points"], dtype=torch.float32) for b in batch],
            num_bins=self.config.encoder.num_bins,
        )
        batch_sdp_orig = [b["semi_dense_points"] for b in batch]
        pack_vertices = False
        if "vertices" in batch[0]:
            batch_vertices = [b["vertices"] for b in batch]
            batch_faces = [b["faces"] for b in batch]
            pack_vertices = True
        for b in batch:
            del b["semi_dense_points"]
            if "vertices" in batch[0]:
                del b["vertices"]
                del b["faces"]
        collated_batch = torch.utils.data.dataloader.default_collate(batch)
        collated_batch["semi_dense_points"] = batch_sdp
        collated_batch["semi_dense_points_orig"] = batch_sdp_orig
        if pack_vertices:
            collated_batch["vertices"] = batch_vertices
            collated_batch["faces"] = batch_faces
        return collated_batch

    @staticmethod
    def move_batch_to_device(batch, device, dtype=torch.float32):
        """Move all tensors (including SparseTensors) to device with optional dtype conversion."""
        for k, v in batch.items():
            if isinstance(v, torchsparse.SparseTensor) or isinstance(v, torch.Tensor):
                if (
                    isinstance(v, torch.Tensor)
                    and v.dtype == torch.float32
                    and v.dtype != dtype
                ):
                    batch[k] = batch[k].to(dtype=dtype)
                batch[k] = batch[k].to(device)
        return batch


def get_boxes_from_masks(rectified_masks):
    """Extract bounding boxes from point projection masks in xyxy format."""
    boxes = []
    for i in range(len(rectified_masks)):
        # get bounds and convert to box in xyxy format
        mask = np.squeeze(rectified_masks[i])
        y, x = np.where(mask > 0.5)
        if len(x) == 0 or len(y) == 0:
            box = [0, 0, mask.shape[1], mask.shape[0]]
        else:
            box = [np.min(x), np.min(y), np.max(x), np.max(y)]
        boxes.append(box)
    return np.array(boxes).reshape(-1, 2, 2)
