# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.


import io
import math
import random

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from preprocessing.helper import plot_dots, rectify_images
from sklearn.cluster import KMeans


def get_image_data_based_on_strategy(
    pkl_sample, num_views, scale, is_rgb, strategy="cluster"
):

    if strategy == "cluster":
        selected_image_data = cluster_and_select_images(
            pkl_sample, num_views, scale, is_rgb
        )
    elif strategy == "last_n":
        selected_image_data = last_n_view_selection(
            pkl_sample, num_views, scale, is_rgb
        )
    elif strategy == "view_angle":
        selected_image_data = view_angle_based_strategy(
            pkl_sample, num_views, scale, is_rgb
        )

    if len(selected_image_data) < num_views:
        # add repeats from the random image data in the list
        selected_image_data = selected_image_data + random.choices(
            selected_image_data, k=num_views - len(selected_image_data)
        )

    (
        rectified_images,
        rectified_masks,
        rectified_camera_params,
    ) = rectify_images(
        torch.from_numpy(np.array([x[1] for x in selected_image_data])),
        torch.from_numpy(np.array([x[2] for x in selected_image_data])).unsqueeze(-1),
        torch.from_numpy(np.array([x[3] for x in selected_image_data])),
    )

    # rotate if SLAM nebula or just RGB
    if pkl_sample.get("is_nebula", False):
        # rotate the image ccw
        rectified_masks = [rectified_masks[i] for i in range(len(rectified_masks))]
        for im_idx in range(len(rectified_masks)):
            rectified_masks[im_idx] = np.rot90(rectified_masks[im_idx], 1)
        rectified_masks = np.stack(rectified_masks)
    rectified_point_masks = [
        np.where(rectified_masks[i] > 0) for i in range(len(rectified_masks))
    ]
    rectified_point_masks = [
        np.stack(
            [
                np.ones_like(rectified_point_masks[i][1]) * i,
                rectified_point_masks[i][1],
                rectified_point_masks[i][0],
            ],
            axis=1,
        )
        for i in range(len(rectified_point_masks))
    ]
    # rectified_camera_params = convert_to_4x4(rectified_camera_params)
    camera_to_worlds = np.stack([x[4] for x in selected_image_data], axis=0)
    if pkl_sample.get("is_nebula", False):
        # rotate the image ccw
        rectified_images = [rectified_images[i] for i in range(len(rectified_images))]
        for im_idx in range(len(rectified_images)):
            rectified_images[im_idx] = np.rot90(rectified_images[im_idx], 1)
            rectified_camera_params[im_idx] = rotate_intrinsics_ccw90(
                rectified_camera_params[im_idx], rectified_images[im_idx].shape[0]
            )
            camera_to_worlds[im_idx] = rotate_extrinsics_ccw90(camera_to_worlds[im_idx])
        rectified_images = np.stack(rectified_images)
    return (
        rectified_images,
        rectified_point_masks,
        rectified_camera_params,
        camera_to_worlds.astype(np.float32),
    )


def convert_to_4x4(camera_params):
    camera_param_4x4 = [np.eye(4) for _ in range(len(camera_params))]
    for i, c in enumerate(camera_params):
        camera_param_4x4[i][:3, :3] = c
    return np.stack(camera_param_4x4)


def rotate_intrinsics_ccw90(cam4x4, new_width):
    new_cam4x4 = cam4x4.copy()
    new_cam4x4[0, 0] = cam4x4[1, 1]
    new_cam4x4[1, 1] = cam4x4[0, 0]
    new_cam4x4[0, 2] = cam4x4[1, 2]
    new_cam4x4[1, 2] = new_width - cam4x4[0, 2]
    return new_cam4x4


def rotate_extrinsics_ccw90(cam4x4):
    R_img = np.zeros((3, 3))
    R_img[0, 1] = -1
    R_img[1, 0] = 1
    R_img[2, 2] = 1
    pre_transform = np.eye(4)
    pre_transform[:3, :3] = R_img
    new_cam4x4 = cam4x4 @ pre_transform
    return new_cam4x4


def cluster_and_select_images(pkl_sample, num_views, scale, is_rgb):
    (
        visible_points_key,
        camera_model_key,
        image_data_key,
        obj_pt_pred_key,
        camera_params_key,
    ) = get_key_names(pkl_sample, is_rgb)
    visible_points_model_counts = np.array(
        [
            (sample_i, x.shape[0])
            for sample_i, x in enumerate(pkl_sample[visible_points_key])
        ]
    )
    camera_centeres = []
    for i in range(len(pkl_sample[camera_model_key])):
        camera_to_world = np.linalg.inv(pkl_sample[camera_model_key][i].numpy())
        camera_to_world[:3, 3] = camera_to_world[:3, 3] * scale
        camera_centeres.append(camera_to_world[:3, 3])
    kmeans_labels = create_k_clusters(camera_centeres, num_views * 2)
    selected_image_indices = []
    for i in range(kmeans_labels.max() + 1):
        argmax = visible_points_model_counts[kmeans_labels == i, 1].argmax()
        visible_points_idx = visible_points_model_counts[kmeans_labels == i, 0][argmax]
        selected_image_indices.append(visible_points_idx)

    selected_image_data = []

    for selected_image_index in selected_image_indices:
        buffer = io.BytesIO(pkl_sample[image_data_key][selected_image_index])
        decoded_image = Image.open(buffer)
        if is_rgb:
            decoded_image = decoded_image.convert("RGB")
        else:
            decoded_image = decoded_image.convert("L")
        image = np.array(decoded_image)
        uv_fisheye = pkl_sample[obj_pt_pred_key][selected_image_index].numpy()
        uv_fisheye_mask = plot_dots(
            uv_fisheye,
            H=image.shape[0],
            W=image.shape[1],
        )
        camera_to_world = np.linalg.inv(
            pkl_sample[camera_model_key][selected_image_index].numpy()
        )
        camera_to_world[:3, 3] = camera_to_world[:3, 3] * scale
        selected_image_data.append(
            (
                selected_image_index,
                image,
                uv_fisheye_mask,
                np.array(pkl_sample[camera_params_key][selected_image_index]),
                camera_to_world,
                get_valid_uv_fisheye(uv_fisheye, image.shape[1], image.shape[0]),
                # u,
                # v,
                # z,
            )
        )

    any_valid_image = False
    for x in selected_image_data:
        if x[5] > 0:
            any_valid_image = True
            break

    if any_valid_image:
        selected_image_data = [x for x in selected_image_data if x[5] > 0]

    selected_image_data = sorted(selected_image_data, key=lambda x: x[5], reverse=True)
    selected_image_data = selected_image_data[:num_views]

    return selected_image_data


def last_n_view_selection(pkl_sample, num_views, scale, is_rgb):
    indices_to_return = list(range(len(pkl_sample["Ts_camera_model"])))[-num_views:]
    selected_image_data = index_to_data(indices_to_return, pkl_sample, scale, is_rgb)
    return selected_image_data


def index_to_data(indices_to_return, pkl_sample, scale, is_rgb):
    (
        visible_points_key,
        camera_model_key,
        image_data_key,
        obj_pt_pred_key,
        camera_params_key,
    ) = get_key_names(pkl_sample, is_rgb)

    selected_image_data = []

    for selected_image_index in indices_to_return:
        buffer = io.BytesIO(pkl_sample[image_data_key][selected_image_index])
        decoded_image = Image.open(buffer)
        if is_rgb:
            decoded_image = decoded_image.convert("RGB")
        else:
            decoded_image = decoded_image.convert("L")
        image = np.array(decoded_image)
        uv_fisheye = pkl_sample[obj_pt_pred_key][selected_image_index].numpy()
        uv_fisheye_mask = plot_dots(
            uv_fisheye,
            H=image.shape[0],
            W=image.shape[1],
        )
        camera_to_world = np.linalg.inv(
            pkl_sample[camera_model_key][selected_image_index].numpy()
        )
        camera_to_world[:3, 3] = camera_to_world[:3, 3] * scale
        selected_image_data.append(
            (
                selected_image_index,
                image,
                uv_fisheye_mask,
                np.array(pkl_sample[camera_params_key][selected_image_index]),
                camera_to_world,
            )
        )
    return selected_image_data


def view_angle_based_strategy(pkl_sample, num_views, scale, is_rgb):
    # greedy strategy to select the best views
    N = num_views * 2
    is_nebula = pkl_sample.get("is_nebula", False)
    camera_centers = (
        torch.linalg.inv(pkl_sample["Ts_camera_model"])[:, :3, 3].cpu().numpy()
    )
    paddedCropsXYWHC = []
    for cam_idx in range(len(camera_centers)):
        uv_fisheye = pkl_sample["object_point_projections"][cam_idx].numpy()
        # get the crop dimensions
        x_min, x_max = np.min(uv_fisheye[:, 0]), np.max(uv_fisheye[:, 0])
        y_min, y_max = np.min(uv_fisheye[:, 1]), np.max(uv_fisheye[:, 1])
        paddedCropsXYWHC.append(
            np.array(
                [
                    x_min,
                    y_min,
                    x_max - x_min,
                    y_max - y_min,
                ]
            )
        )
    paddedCropsXYWHC = np.stack(paddedCropsXYWHC)
    region_indices = []
    object_in_good_view = []
    for cam_idx, camera_center in enumerate(camera_centers):
        region_indices.append(
            hemisphere_region(camera_center[0], camera_center[1], camera_center[2], N)
        )
        object_in_good_view.append(
            check_object_in_good_view(paddedCropsXYWHC[cam_idx], is_nebula)
        )

    # 0: not occupied, 1: occupied but bad view, 2: occupied and good view
    region_views_occupied = [0] * N
    region_view_indices = [None] * N

    for cam_idx in range(len(pkl_sample["Ts_camera_model"])):
        if (
            region_indices[cam_idx] != -1
            and region_views_occupied[region_indices[cam_idx]] < 2
            and object_in_good_view[cam_idx] != 0
        ):
            # only replace if the region is not occupied or occupied with bad view and new view is good view
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

    selected_image_data = index_to_data(indices_to_return, pkl_sample, scale, is_rgb)

    return selected_image_data


def get_key_names(pkl_sample, is_rgb):
    visible_points_key = (
        "rgb_visible_points_model"
        if is_rgb and "rgb_visible_points_model" in pkl_sample
        else "visible_points_model"
    )
    camera_model_key = (
        "Ts_rgbCamera_model"
        if is_rgb and "Ts_rgbCamera_model" in pkl_sample
        else "Ts_camera_model"
    )
    image_data_key = (
        "rgb_image_data" if is_rgb and "rgb_image_data" in pkl_sample else "image_data"
    )
    obj_pt_pred_key = (
        "rgb_object_point_projections"
        if is_rgb and "rgb_object_point_projections" in pkl_sample
        else "object_point_projections"
    )
    camera_params_key = (
        "rgb_camera_params"
        if is_rgb and "rgb_camera_params" in pkl_sample
        else "camera_params"
    )
    return (
        visible_points_key,
        camera_model_key,
        image_data_key,
        obj_pt_pred_key,
        camera_params_key,
    )


def create_k_clusters(centers, k):
    kmeans = KMeans(n_clusters=min(k, len(centers)), random_state=42, n_init="auto")
    kmeans.fit(centers)
    labels = kmeans.labels_  # Cluster labels for each point
    return labels


def get_valid_uv_fisheye(feye, width, height):
    pad_u, pad_v = int(width * 0.2), int(height * 0.2)
    mask_u = np.where(np.logical_and(feye[:, 0] > pad_u, feye[:, 0] < width - pad_u))[0]
    mask_v = np.where(np.logical_and(feye[:, 1] > pad_v, feye[:, 1] < height - pad_v))[
        0
    ]
    mask = np.intersect1d(mask_u, mask_v)
    return feye[mask].shape[0]


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


def check_object_in_good_view(xywh, isNebula):
    if isNebula:
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
            return 2 + xywh[2] * xywh[3] / (H * W)
        return 1
    return 0


def crop_pad_preselected_views_with_background(
    view_imgs, view_msks_uvs, view_int, crop_size, add_point_locations
):
    selected_point_uvs = [
        view_msks_uvs[v_idx][:, 1:] for v_idx in range(len(view_imgs))
    ]

    selected_view_point_msks = [
        plot_dots(
            selected_point_uvs[v_idx],
            H=view_imgs[0].shape[0],
            W=view_imgs[0].shape[1],
        )
        for v_idx in range(len(selected_point_uvs))
    ]

    selected_view_imgs = [None for _ in range(len(view_imgs))]
    selected_view_uncropped_imgs = [None for _ in range(len(view_imgs))]
    selected_view_int = [None for _ in range(len(view_imgs))]
    for v_idx in range(len(view_imgs)):
        (
            selected_view_imgs[v_idx],
            selected_view_uncropped_imgs[v_idx],
            selected_view_point_msks[v_idx],
            selected_view_int[v_idx],
        ) = crop_and_resize(
            view_imgs[v_idx],
            selected_view_point_msks[v_idx],
            camera_intrinsics=view_int[v_idx],
            target_size=crop_size,
            resize_mode="bilinear",
            do_padding=False,
            mask_uncropped=None,
            mask_m2f=None,
        )

    kernel = np.ones((3, 3), np.uint8)
    for v_idx in range(len(selected_view_point_msks)):
        selected_view_point_msks[v_idx] = cv2.dilate(
            selected_view_point_msks[v_idx], kernel, iterations=1
        )

    selected_view_int = np.stack(selected_view_int)
    selected_view_imgs = np.stack(selected_view_imgs)
    selected_view_point_msks = np.stack(selected_view_point_msks)
    selected_view_imgs = selected_view_imgs.astype(np.float32) / 255.0
    selected_view_point_msks = selected_view_point_msks.astype(np.float32) / 255.0
    if len(selected_view_imgs.shape) == 3:
        selected_view_imgs = np.repeat(
            selected_view_imgs[:, np.newaxis, ...], 3, axis=1
        )
    else:
        selected_view_imgs = selected_view_imgs.transpose((0, 3, 1, 2))
    if add_point_locations:
        selected_view_imgs[:, 1, ...] = (
            selected_view_imgs[:, 1, ...] * (1 - selected_view_point_msks)
            + selected_view_point_msks
        )

    return (
        selected_view_imgs.astype(np.float32),
        selected_view_uncropped_imgs,
        selected_view_point_msks.astype(np.float32),
        selected_view_int.astype(np.float32),
    )


def crop_and_resize(
    image,
    mask,
    camera_intrinsics=None,
    target_size: int = 448,
    resize_mode: str = "bilinear",
    do_padding: bool = True,
    mask_uncropped=None,
    image_for_box=None,
    mask_m2f=None,
    pad_value=20,
):
    if camera_intrinsics is not None:
        camera_intrinsics = camera_intrinsics.copy()
    if image_for_box is None:
        image_for_box = image
    # image is a numpy array in the range [0, 255], uint8
    # mask is a numpy array of bools
    image_pil = Image.fromarray(image)
    image_pil_for_box = Image.fromarray(image_for_box)
    mask_pil = Image.fromarray(mask)
    # Step 1: Find the bounding box coordinates from the binary mask
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        x_min, x_max = 0, 200
        y_min, y_max = 0, 200
    else:
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        diff_x = x_max - x_min
        diff_y = y_max - y_min
        if diff_x > diff_y:
            y_mid = (y_min + y_max) // 2
            y_min = y_mid - diff_x // 2
            y_max = y_mid + diff_x // 2
            if y_min < 0:
                y_max += -y_min
                y_min = 0
            elif y_max > image.shape[0] - 1:
                y_min -= y_max - (image.shape[0] - 1)
                y_max = image.shape[0] - 1
        else:
            x_mid = (x_min + x_max) // 2
            x_min = x_mid - diff_y // 2
            x_max = x_mid + diff_y // 2
            if x_min < 0:
                x_max += -x_min
                x_min = 0
            elif x_max > image.shape[1] - 1:
                x_min -= x_max - (image.shape[1] - 1)
                x_max = image.shape[1] - 1
        # if use_resize_adjusted_pad_value:
        # pad_value = max(10, int(pad_value * (x_max - x_min) / target_size))
        x_min, x_max = (
            max(0, x_min - pad_value),
            min(image.shape[1] - 1, x_max + pad_value),
        )
        y_min, y_max = (
            max(0, y_min - pad_value),
            min(image.shape[0] - 1, y_max + pad_value),
        )

    if camera_intrinsics is not None:
        camera_intrinsics[0, 2] -= x_min
        camera_intrinsics[1, 2] -= y_min

    # Create a copy of the image
    img_with_box = image_pil_for_box.copy().convert("RGB")
    if mask_uncropped is not None:
        img_with_box = Image.fromarray(
            np.array(img_with_box) * (mask_uncropped[:, :, None] > 0).astype(np.uint8)
        )
    # Draw a red box on the copied image
    draw = ImageDraw.Draw(img_with_box)
    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
    img_with_box = img_with_box.transpose(Image.ROTATE_270)

    # Crop the image using these bounds
    image_cropped = image_pil.crop((x_min, y_min, x_max + 1, y_max + 1))
    mask_cropped = mask_pil.crop((x_min, y_min, x_max + 1, y_max + 1))
    if mask_m2f is not None:
        mask_m2f_cropped = Image.fromarray(mask_m2f).crop(
            (x_min, y_min, x_max + 1, y_max + 1)
        )
        mask_m2f = (np.array(mask_m2f_cropped) > 127).astype(np.uint8)
        mask_cropped = Image.fromarray(np.array(mask_cropped) * mask_m2f)

    # Step 2: Resize the cropped image so that the long edge is 200 pixels
    width, height = image_cropped.size
    resize_size = int(target_size * 0.90) if do_padding else target_size
    if width > height:
        new_width = resize_size
        new_height = max(int((height / width) * resize_size), 8)
    else:
        new_height = resize_size
        new_width = max(int((width / height) * resize_size), 8)

    sx = new_width / width
    sy = new_height / height
    if camera_intrinsics is not None:
        camera_intrinsics[0, 0] *= sx
        camera_intrinsics[1, 1] *= sy
        camera_intrinsics[0, 2] *= sx
        camera_intrinsics[1, 2] *= sy

    resize_mode = Image.LANCZOS if resize_mode == "bilinear" else Image.NEAREST
    image_resized = image_cropped.resize((new_width, new_height), resize_mode)
    mask_resized = mask_cropped.resize((new_width, new_height), Image.NEAREST)

    # Step 3: Pad the image to reach 448x448, centering the content

    delta_w = target_size - new_width
    delta_h = target_size - new_height
    padding = (
        delta_w // 2,
        delta_h // 2,
        delta_w - (delta_w // 2),
        delta_h - (delta_h // 2),
    )

    if camera_intrinsics is not None:
        camera_intrinsics[0, 2] += padding[0]
        camera_intrinsics[1, 2] += padding[1]

    image_resized = ImageOps.expand(image_resized, padding, fill=0)
    mask_resized = ImageOps.expand(mask_resized, padding, fill=0)

    return (
        np.array(image_resized),
        img_with_box,
        np.array(mask_resized),
        camera_intrinsics,
    )
