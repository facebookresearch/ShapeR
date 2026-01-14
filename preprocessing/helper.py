# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Preprocessing utilities for ShapeR.

Includes fisheye rectification, point cloud processing, camera projection,
and image cropping/resizing functions.
"""

from collections import OrderedDict

import einops
import numpy as np
import torch
import torchsparse
from torchsparse.utils.collate import sparse_collate

from .camera import CameraTW, param_to_matrix, rectify_video

from .point_cloud import PointCloud


def get_caption(data):
    try:
        return (
            data["category"] + "," + data["openVocLabel"] + "," + data["vlmDescription"]
        )
    except Exception as e:
        return "3D object"


def get_parameters_from_state_dict(state_dict, filter_key):
    """Extract parameters matching a key prefix from a state dict."""
    new_state_dict = OrderedDict()
    for k in state_dict:
        if k.startswith(filter_key):
            new_state_dict[k.replace(filter_key + ".", "", 1)] = state_dict[k]
    return new_state_dict


def preprocess_point_cloud(
    point_cloud_batch, num_bins, object_bboxes=None, push_to_positive_quadrant=False
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
            coords=point_cloud.coords.int(),
            feats=point_cloud.points.float(),
        )

        pc_sparse_tensors.append(pc_sparse_tensor)

    pc_sparse_tensor_batch = sparse_collate(pc_sparse_tensors)

    return pc_sparse_tensor_batch


def crop_and_resize(
    image,
    mask,
    camera_intrinsics,
    target_size=448,
):
    """Crop image around mask bounding box and resize to target size, updating intrinsics."""
    camera_intrinsics = camera_intrinsics.clone()
    cropped_images = []
    for image_idx in range(image.shape[0]):
        y_indices, x_indices = torch.where(mask[image_idx] > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            x_min, x_max = 0, 200
            y_min, y_max = 0, 200
        else:
            x_min, x_max = torch.min(x_indices), torch.max(x_indices)
            y_min, y_max = torch.min(y_indices), torch.max(y_indices)
            diff_x = x_max - x_min
            diff_y = y_max - y_min
            if diff_x > diff_y:
                y_mid = (y_min + y_max) // 2
                y_min = y_mid - diff_x // 2
                y_max = y_mid + diff_x // 2
                if y_min < 0:
                    y_max += -y_min
                    y_min = 0
                elif y_max > image[image_idx].shape[0] - 1:
                    y_min -= y_max - (image[image_idx].shape[0] - 1)
                    y_max = image[image_idx].shape[0] - 1
            else:
                x_mid = (x_min + x_max) // 2
                x_min = x_mid - diff_y // 2
                x_max = x_mid + diff_y // 2
                if x_min < 0:
                    x_max += -x_min
                    x_min = 0
                elif x_max > image[image_idx].shape[1] - 1:
                    x_min -= x_max - (image[image_idx].shape[1] - 1)
                    x_max = image[image_idx].shape[1] - 1
            x_min, x_max = (
                max(0, x_min - 20),
                min(image[image_idx].shape[1] - 1, x_max + 20),
            )
            y_min, y_max = (
                max(0, y_min - 20),
                min(image[image_idx].shape[0] - 1, y_max + 20),
            )

        camera_intrinsics[image_idx, 0, 2] -= x_min
        camera_intrinsics[image_idx, 1, 2] -= y_min
        image_cropped = image[image_idx][y_min : y_max + 1, x_min : x_max + 1]

        # Step 2: Resize the cropped image so that the long edge is 200 pixels
        height, width = image_cropped.shape
        if width > height:
            new_width = target_size
            new_height = max(int((height / width) * target_size), 8)
        else:
            new_height = target_size
            new_width = max(int((width / height) * target_size), 8)

        sx = new_width / width
        sy = new_height / height

        camera_intrinsics[image_idx, 0, 0] *= sx
        camera_intrinsics[image_idx, 1, 1] *= sy
        camera_intrinsics[image_idx, 0, 2] *= sx
        camera_intrinsics[image_idx, 1, 2] *= sy

        image_cropped = torch.nn.functional.interpolate(
            image_cropped.float().unsqueeze(0).unsqueeze(0) / 255.0,
            size=(new_height, new_width),
            mode="bilinear",
        )
        image_cropped = image_cropped.squeeze(0).repeat((3, 1, 1))

        # Step 3: Pad the image to reach 448x448 or 228x228, centering the content

        delta_w = target_size - new_width
        delta_h = target_size - new_height
        padding = (
            delta_w // 2,
            delta_h // 2,
            delta_w - (delta_w // 2),
            delta_h - (delta_h // 2),
        )

        image_padded = torch.zeros(
            (image_cropped.shape[0], target_size, target_size),
            dtype=image_cropped.dtype,
            device=image_cropped.device,
        )
        image_padded[
            :,
            padding[1] : image_padded.shape[1] - padding[3],
            padding[0] : image_padded.shape[2] - padding[2],
        ] = image_cropped

        camera_intrinsics[image_idx, 0, 2] += padding[0]
        camera_intrinsics[image_idx, 1, 2] += padding[1]

        cropped_images.append(image_padded)
    return torch.stack(cropped_images, dim=0), camera_intrinsics


def pad_for_rectification(crops, masks, paddedCropsXYWHC, is_ariagen2):
    """Pad cropped images back to full frame size for rectification."""
    if is_ariagen2:
        H, W = 512, 512
    else:
        H, W = 480, 640
    pad_crop = torch.zeros(
        (len(crops), H, W, 1), dtype=crops[0].dtype, device=crops[0].device
    )
    pad_mask = torch.zeros(
        (len(masks), H, W, 1), dtype=masks[0].dtype, device=masks[0].device
    )

    for c_idx in range(len(crops)):
        xywh = paddedCropsXYWHC[c_idx]
        pad_crop[c_idx, xywh[1] : xywh[1] + xywh[3], xywh[0] : xywh[0] + xywh[2], :] = (
            crops[c_idx]
        )
        pad_mask[c_idx, xywh[1] : xywh[1] + xywh[3], xywh[0] : xywh[0] + xywh[2], :] = (
            masks[c_idx]
        )

    return pad_crop, pad_mask


def rectify_images(images, masks, camera_params):
    """Rectify fisheye images to pinhole projection."""
    rectified_images = []
    rectified_masks = []
    rectified_camera_params = []
    for _, (image, mask, camera_param) in enumerate(zip(images, masks, camera_params)):
        if image.ndim == 2:
            video = image[None, ..., None]
        elif image.ndim == 3:
            video = image[None, ...]
        else:
            raise ValueError(f"Unknown image shape {image.shape}")
        width, height = video.shape[2], video.shape[1]

        video = einops.rearrange(video, "b h w c -> b c h w").float() / 255

        # assumes mask's last channel is 1
        video_mask = mask[None, ...]
        video_mask = einops.rearrange(video_mask, "b h w c -> b c h w").float()
        cam = CameraTW.from_surreal(
            width=width,
            height=height,
            params=camera_param,
            type_str="Fisheye624",
        ).unsqueeze(0)
        vid_rectified, cam_rectified = rectify_video(video, cam, pinhole_fxy_factor=1.0)
        vid_mask_rectified, _ = rectify_video(
            video_mask, cam, pinhole_fxy_factor=1.0, interp_mode="nearest"
        )

        vid_rectified = (
            einops.rearrange(vid_rectified, "b c h w -> b h w c").clamp(0, 1) * 255
        ).to(torch.uint8)

        vid_mask_rectified = (
            einops.rearrange(vid_mask_rectified, "b c h w -> b h w c").clamp(0, 1) * 255
        ).to(torch.uint8)

        rectified_images.append(vid_rectified[0, :, :].squeeze(-1))
        rectified_masks.append(vid_mask_rectified[0, :, :, 0])
        rectified_camera_params.append(param_to_matrix(cam_rectified.params[0]))

    return (
        torch.stack(rectified_images, dim=0).numpy(),
        torch.stack(rectified_masks, dim=0).numpy(),
        torch.stack(rectified_camera_params, dim=0).numpy(),
    )


def rotate_intrinsics_ccw90(cam4x4, new_width):
    """Rotate camera intrinsics for 90-degree CCW image rotation."""
    new_cam4x4 = cam4x4.clone()
    new_cam4x4[0, 0] = cam4x4[1, 1]
    new_cam4x4[1, 1] = cam4x4[0, 0]
    new_cam4x4[0, 2] = cam4x4[1, 2]
    new_cam4x4[1, 2] = new_width - cam4x4[0, 2]
    return new_cam4x4


def rotate_extrinsics_ccw90(cam4x4):
    """Rotate camera extrinsics for 90-degree CCW image rotation."""
    R_img = torch.zeros((3, 3), device=cam4x4.device)
    R_img[0, 1] = -1
    R_img[1, 0] = 1
    R_img[2, 2] = 1
    pre_transform = torch.eye(4, device=cam4x4.device)
    pre_transform[:3, :3] = R_img
    new_cam4x4 = cam4x4 @ pre_transform
    return new_cam4x4


def project_point_to_image(points_world, camera_intrinsics, cam2world, W, H):
    K = camera_intrinsics[:3, :3]
    world2cam = np.linalg.inv(cam2world)
    points_camera = (world2cam[:3, :3] @ points_world.T).T + world2cam[:3, 3]
    points_image = (K @ points_camera.T).T
    points_image = points_image / points_image[:, 2:3]
    return plot_dots(points_image[:, :2], W, H)


def plot_dots(uv, W, H):
    """Render 2D point projections as a binary mask image."""
    # Initialize a blank image
    img = np.zeros((H, W), dtype=np.float32)

    # Create a 2D histogram of the points
    x = np.clip(uv[:, 0], 0, W - 1).astype(int)
    y = np.clip(uv[:, 1], 0, H - 1).astype(int)
    np.add.at(img, (y, x), 1)

    return (img * 255).astype(np.uint8)


def sign_plus(x):
    sgn = np.ones_like(x)
    sgn[x < 0.0] = -1.0
    return sgn


def project_point_to_image_with_distortion(
    points_zup, T_camera_model, camera_params, image_dims
):
    """Project 3D points to image using FISHEYE624 distortion model."""
    eps = 1e-9
    P = points_zup.shape[0]
    ones = np.ones((P, 1), dtype=points_zup.dtype)
    points_zup_4 = np.concatenate([points_zup, ones], axis=1)

    points_out = (T_camera_model @ points_zup_4.T).T

    denom = points_out[..., 3:]  # denominator
    points_camera = points_out[..., :3] / (denom + eps)
    xyz = points_camera[None]
    params = camera_params[None]
    B, N = xyz.shape[0], xyz.shape[1]

    # Radial correction.
    z = xyz[:, :, 2].reshape(B, N, 1)
    # Do not use torch.sign(z) it leads to 0.0 zs if z == 0.0 which leads to a
    # nan when we compute xy/z
    z = np.where(np.abs(z) < eps, eps * sign_plus(z), z)
    ab = xyz[:, :, :2] / z
    # make sure abs are not too small or 0 otherwise gradients are nan
    ab = np.where(np.abs(ab) < eps, eps * sign_plus(ab), ab)
    r = np.linalg.norm(ab, axis=-1, ord=2)[:, :, np.newaxis]
    th = np.arctan(r)
    th_divr = np.where(r < eps, np.ones_like(ab), ab / r)
    th_k = th.reshape(B, N, 1).copy()
    for i in range(6):
        th_k = th_k + params[:, -12 + i].reshape(B, 1, 1) * np.power(th, 3 + i * 2)
    xr_yr = th_k * th_divr
    uv_dist = xr_yr

    # Tangential correction.
    p0 = params[:, -6].reshape(B, 1)
    p1 = params[:, -5].reshape(B, 1)
    xr = xr_yr[:, :, 0].reshape(B, N)
    yr = xr_yr[:, :, 1].reshape(B, N)
    xr_yr_sq = np.square(xr_yr)
    xr_sq = xr_yr_sq[:, :, 0].reshape(B, N)
    yr_sq = xr_yr_sq[:, :, 1].reshape(B, N)
    rd_sq = xr_sq + yr_sq
    uv_dist_tu = uv_dist[:, :, 0] + ((2.0 * xr_sq + rd_sq) * p0 + 2.0 * xr * yr * p1)
    uv_dist_tv = uv_dist[:, :, 1] + ((2.0 * yr_sq + rd_sq) * p1 + 2.0 * xr * yr * p0)
    uv_dist = np.stack([uv_dist_tu, uv_dist_tv], axis=-1)  # Avoids in-place complaint.

    # Thin Prism correction.
    s0 = params[:, -4].reshape(B, 1)
    s1 = params[:, -3].reshape(B, 1)
    s2 = params[:, -2].reshape(B, 1)
    s3 = params[:, -1].reshape(B, 1)
    rd_4 = np.square(rd_sq)
    uv_dist[:, :, 0] = uv_dist[:, :, 0] + (s0 * rd_sq + s1 * rd_4)
    uv_dist[:, :, 1] = uv_dist[:, :, 1] + (s2 * rd_sq + s3 * rd_4)

    # Finally, apply standard terms: focal length and camera centers.
    if params.shape[-1] == 15:
        fx_fy = params[:, 0].reshape(B, 1, 1)
        cx_cy = params[:, 1:3].reshape(B, 1, 2)
    else:
        fx_fy = params[:, 0:2].reshape(B, 1, 2)
        cx_cy = params[:, 2:4].reshape(B, 1, 2)
    uv = (uv_dist * fx_fy + cx_cy)[0]

    valid_depth = points_camera[..., 2] > 0
    valid = (
        (uv[..., 0] >= 0)
        & (uv[..., 1] >= 0)
        & (uv[..., 0] < image_dims[1])
        & (uv[..., 1] < image_dims[0])
        & valid_depth
    )
    uv_fisheye_mask = plot_dots(
        uv[valid],
        H=image_dims[0],
        W=image_dims[1],
    )
    return uv_fisheye_mask
