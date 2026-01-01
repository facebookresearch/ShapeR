# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch


def batched_rays_from_cameras(camera_to_world, camera_intrinsics, H, W, scene_radius):
    orig_dtype = camera_to_world.dtype
    func_dtype = torch.float32
    camc = camera_to_world[:, :3, 3].to(func_dtype)  # (B, 3)
    # FIXME: adjust near/far for each camera separately instead
    cam_to_scene_dist = (camc).norm(dim=-1)
    far, near = cam_to_scene_dist + scene_radius, cam_to_scene_dist - scene_radius
    near = near.mean().clamp(0.01).item()
    far = far.mean().item()

    u, v = torch.meshgrid(
        torch.arange(W, device=camera_to_world.device),
        torch.arange(H, device=camera_to_world.device),
        indexing="xy",
    )
    # Flatten the tensors
    u, v = u.flatten(), v.flatten()
    uv = torch.stack(
        [u, v, torch.ones_like(u), torch.ones_like(u)], dim=-1
    ).float()  # (H*W, 4)

    ray_dirs = []
    for i in range(camera_to_world.shape[0]):
        XYZ = (
            camera_to_world[i].to(func_dtype)
            @ torch.linalg.inv(camera_intrinsics[i].to(func_dtype))
            @ uv.T
        )  # (4, H*W)
        XYZ = XYZ / XYZ[-1:, :]  # (4, H*W)
        XYZ = XYZ[:3, :].T  # (H*W, 3)
        ray_dir = torch.nn.functional.normalize(XYZ - camc[i, None, :], p=2, dim=-1)
        ray_dirs.append(ray_dir)

    ray_dirs = torch.stack(ray_dirs, dim=0)  # (B, H*W, 3)
    ray_origins = camc[:, None, :].expand(-1, H * W, -1)  # (B, H*W, 3)
    ray_nears = near * torch.ones_like(ray_origins[..., 0])  # (B, H*W)
    ray_fars = far * torch.ones_like(ray_nears)  # (B, H*W)
    return (
        ray_origins.to(orig_dtype),
        ray_dirs.to(orig_dtype),
        ray_nears.to(orig_dtype),
        ray_fars.to(orig_dtype),
    )


def get_image_ray_plucker(
    camera_to_world,
    camera_intrinsics,
    H,
    W,
):
    B = camera_to_world.shape[0]
    ray_bundle = batched_rays_from_cameras(
        camera_to_world, camera_intrinsics, H, W, 1.6
    )
    plucker = ray_origin_dir_to_plucker_coords(
        ray_bundle[0].reshape(B, H, W, 3),
        ray_bundle[1].reshape(B, H, W, 3),
    )
    return plucker


def ray_origin_dir_to_plucker_coords(cam_pos: torch.Tensor, ray_dirs: torch.Tensor):
    """
    Args:
        cam_pos (torch.Tensor): (..., 3)
        ray_dirs (torch.Tensor): (..., 3)
    """
    cross = torch.cross(cam_pos, ray_dirs, dim=-1)
    plucker = torch.cat((ray_dirs, cross), dim=-1)
    return plucker
