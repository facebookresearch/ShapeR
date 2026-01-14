# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch


def hemisphere_region(x, y, z, num_regions):
    phi = math.atan2(y, x)
    if phi < 0:
        phi += 2 * math.pi

    sector_width = math.pi * 2 / num_regions
    sector = int(phi // sector_width)  # yields 0..num_regions-1
    if sector < 0 or sector > num_regions - 1:
        return -1
    # remap so that adjacent sectors are not next to each other
    remap = [i * 2 for i in range(num_regions // 2)] + [
        i * 2 + 1 for i in range(num_regions // 2)
    ]
    return remap[sector]


def check_object_in_good_view(xywh, is_ariagen2):
    if is_ariagen2:
        H, W = 512, 512
    else:
        H, W = 480, 640
    # object is in good view if it is in the middle 80% of the image
    if (
        xywh[0] > W * 0.20
        and xywh[0] + xywh[2] < W * 0.80
        and xywh[1] > H * 0.20
        and xywh[1] + xywh[3] < H * 0.80
    ):
        # if the object is not too small or too big return true, else false
        if xywh[2] * xywh[3] > 0.025 * H * W and xywh[2] * xywh[3] < 0.75 * H * W:
            # return a score that is 2 + proportion of the object in the image but the added value is between 0 and 1
            return 2 + (xywh[2] * xywh[3] / (H * W)).item()
        return 1
    return 0


def dummy_view_selection_strategy(
    crops, masks, camera_params, Ts_camera_model, paddedCropsXYWHC, N, is_ariagen2
):
    # return last N items
    return (
        crops[-N:],
        masks[-N:],
        camera_params[-N:],
        Ts_camera_model[-N:],
        paddedCropsXYWHC[-N:],
    )


def view_angle_based_strategy(
    crops, masks, camera_params, Ts_camera_model, paddedCropsXYWHC, num_views, is_ariagen2
):
    N = num_views * 2
    camera_centers = torch.linalg.inv(Ts_camera_model)[:, :3, 3].cpu().numpy()
    region_indices = []
    object_in_good_view = []
    for cam_idx, camera_center in enumerate(camera_centers):
        region_indices.append(
            hemisphere_region(camera_center[0], camera_center[1], camera_center[2], N)
        )
        object_in_good_view.append(
            check_object_in_good_view(paddedCropsXYWHC[cam_idx], is_ariagen2)
        )
    region_views_occupied = [False] * N
    region_view_indices = [None] * N

    for cam_idx in range(len(crops)):
        if (
            region_indices[cam_idx] != -1
            and region_views_occupied[region_indices[cam_idx]] < 2
            and object_in_good_view[cam_idx] != 0
        ):
            if (
                object_in_good_view[cam_idx]
                > region_views_occupied[region_indices[cam_idx]]
            ):
                region_views_occupied[region_indices[cam_idx]] = object_in_good_view[
                    cam_idx
                ]
                region_view_indices[region_indices[cam_idx]] = cam_idx

    # indices to return = first good views,
    # then remaining bad views so that total is num_views
    # might happen we have less, that is fine
    indices_to_return = []
    for i in range(N):
        if region_views_occupied[i] >= 2:
            indices_to_return.append(region_view_indices[i])

    for i in range(N):
        if region_views_occupied[i] == 1:
            indices_to_return.append(region_view_indices[i])

    indices_to_return = indices_to_return[:num_views]

    # return last N items
    return (
        [crops[i] for i in indices_to_return],
        [masks[i] for i in indices_to_return],
        camera_params[indices_to_return],
        Ts_camera_model[indices_to_return],
        paddedCropsXYWHC[indices_to_return],
    )
