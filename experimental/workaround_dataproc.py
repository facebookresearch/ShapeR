#!/usr/bin/env python3
"""
Depth-Anything-V3 to ShapeR Data Processor

This script processes single-view images using Depth-Anything-V3 for metric depth
estimation and converts the output into the pickle format expected by ShapeR
(https://github.com/facebookresearch/ShapeR/) for single-view 3D reconstruction.

Pipeline:
    1. Run Depth-Anything-V3 inference to get metric depth + camera parameters
    2. Unproject depth to 3D point cloud
    3. Use foreground mask to isolate object points
    4. Use XY-plane mask to estimate ground plane and align to Z-up coordinate system
    5. Center and normalize the point cloud
    6. Apply DBSCAN filtering and FPS downsampling
    7. Export to ShapeR-compatible pickle format

Requirements:
    - depth_anything_3
    - fpsample
    - scikit-learn
    - PIL/Pillow
    - torch
    - numpy

Usage:
    python workaround_dataproc.py

    Expects in ./example/:
        - cup_painting.jpg    (input image)
        - foreground.png      (binary mask of object)
        - xy_plane.png        (binary mask of ground/table plane)
        - caption.txt         (text description of object)
"""

import io
import pickle
from pathlib import Path

import fpsample
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor

from depth_anything_3.api import DepthAnything3


# =============================================================================
# Constants
# =============================================================================

# Transform from Y-down (OpenCV convention) to Z-up (ShapeR convention)
T_Y_DOWN_TO_Z_UP = np.array([
    [1,  0,  0, 0],
    [0,  0,  1, 0],
    [0, -1,  0, 0],
    [0,  0,  0, 1],
], dtype=np.float64)


# =============================================================================
# Geometry Utilities
# =============================================================================

def to_homogeneous_44(ext: np.ndarray) -> np.ndarray:
    """
    Convert (N, 3, 4) extrinsics to (N, 4, 4) homogeneous matrices.

    Args:
        ext: Extrinsic matrices of shape (N, 3, 4) or (N, 4, 4)

    Returns:
        Homogeneous matrices of shape (N, 4, 4)
    """
    if ext.shape[-2:] == (4, 4):
        return ext
    N = ext.shape[0]
    out = np.zeros((N, 4, 4), dtype=ext.dtype)
    out[:, :3, :4] = ext
    out[:, 3, 3] = 1.0
    return out


def world2cam_to_cam2world(world2cam: np.ndarray) -> np.ndarray:
    """
    Invert world-to-camera matrices to get camera-to-world matrices.

    Args:
        world2cam: World-to-camera matrices of shape (N, 4, 4)

    Returns:
        Camera-to-world matrices of shape (N, 4, 4)
    """
    cam2worlds = []
    for idx in range(len(world2cam)):
        cam2worlds.append(np.linalg.inv(world2cam[idx])[None, :, :])
    return np.concatenate(cam2worlds, axis=0)


def apply_transform_to_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Apply a 4x4 homogeneous transform to a point cloud.

    Args:
        points: Point cloud of shape (N, 3)
        transform: 4x4 transformation matrix

    Returns:
        Transformed points of shape (N, 3)
    """
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack([points, ones])
    points_h = (transform @ points_h.T).T
    return points_h[:, :3]


def center_box(points: np.ndarray) -> tuple:
    """
    Center a point cloud at the origin using bounding box center.

    Args:
        points: Point cloud of shape (N, 3)

    Returns:
        Tuple of (centered_points, centering_transform)
    """
    p_min = np.min(points, axis=0)
    p_max = np.max(points, axis=0)
    centroid = (p_min + p_max) / 2
    points_centered = points - centroid

    T = np.eye(4)
    T[:3, 3] = -centroid
    return points_centered, T


# =============================================================================
# Depth Processing
# =============================================================================

def merge_depth_maps_to_pointcloud(
    depths: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    images: np.ndarray = None,
    conf: np.ndarray = None,
    conf_thresh: float = 0.0,
    remove_bottom_percentile: float = 15.0,
) -> np.ndarray:
    """
    Merge multiple depth maps into a single world-space point cloud.

    Args:
        depths: Depth maps of shape (N, H, W)
        intrinsics: Camera intrinsics of shape (N, 3, 3)
        extrinsics: World-to-camera matrices of shape (N, 3, 4) or (N, 4, 4)
        images: Optional RGB images of shape (N, H, W, 3)
        conf: Optional confidence maps of shape (N, H, W)
        conf_thresh: Minimum confidence threshold
        remove_bottom_percentile: Remove bottom X% of points by confidence

    Returns:
        Merged point cloud of shape (M, 3)
    """
    N, H, W = depths.shape
    extrinsics = to_homogeneous_44(extrinsics)

    # Create pixel coordinate grid
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(us)
    pix = np.stack([us, vs, ones], axis=-1).reshape(-1, 3)

    pts_all, conf_all = [], []

    for i in range(N):
        d = depths[i]

        # Build validity mask
        valid = np.isfinite(d) & (d > 0)
        if conf is not None:
            valid &= conf[i] >= conf_thresh
        if not np.any(valid):
            continue

        d_flat = d.reshape(-1)
        vidx = np.flatnonzero(valid.reshape(-1))

        # Unproject to camera space then transform to world space
        K_inv = np.linalg.inv(intrinsics[i])
        c2w = np.linalg.inv(extrinsics[i])

        rays = K_inv @ pix[vidx].T
        Xc = rays * d_flat[vidx][None, :]
        Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))])
        Xw = (c2w @ Xc_h)[:3].T.astype(np.float32)

        pts_all.append(Xw)

        if conf is not None:
            conf_all.append(conf[i].reshape(-1)[vidx])

    if len(pts_all) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    points = np.concatenate(pts_all, axis=0)

    # Remove bottom percentile by confidence
    if conf is not None and remove_bottom_percentile > 0 and len(conf_all) > 0:
        conf_merged = np.concatenate(conf_all, axis=0)
        percentile_thresh = np.percentile(conf_merged, remove_bottom_percentile)
        keep_mask = conf_merged >= percentile_thresh
        points = points[keep_mask]

    return points


# =============================================================================
# Plane Alignment
# =============================================================================

def align_to_xy_plane(
    all_points: np.ndarray,
    xy_points_noisy: np.ndarray,
    target_normal: tuple = (0, 1, 0),
) -> tuple:
    """
    Align point cloud so that a reference plane becomes horizontal (Y=0).

    Uses RANSAC to robustly fit a plane to noisy reference points, then computes
    a rotation to align the plane normal with the target normal.

    Args:
        all_points: Full point cloud to transform, shape (N, 3)
        xy_points_noisy: Points on the reference plane (e.g., table surface), shape (M, 3)
        target_normal: Desired plane normal after alignment (default: Y-up)

    Returns:
        Tuple of (aligned_points, alignment_transform)
    """
    # Fit plane using RANSAC: y = c0*x + c1*z + intercept
    X_z = xy_points_noisy[:, [0, 2]]  # Features: X and Z
    y_h = xy_points_noisy[:, 1]        # Target: Y

    ransac = RANSACRegressor(min_samples=3, residual_threshold=0.01, random_state=42)
    ransac.fit(X_z, y_h)

    # Extract plane normal from coefficients
    # Plane equation: c0*x - 1*y + c1*z + intercept = 0
    c0, c1 = ransac.estimator_.coef_
    normal = np.array([c0, -1, c1])
    normal = normal / np.linalg.norm(normal)

    # Find centroid of inlier points
    inlier_mask = ransac.inlier_mask_
    inlier_points = xy_points_noisy[inlier_mask]
    centroid = np.mean(inlier_points, axis=0)

    # Compute rotation to align normal with target
    target_normal = np.array(target_normal)
    target_normal = target_normal / np.linalg.norm(target_normal)

    # Ensure normal points in same hemisphere as target
    if np.dot(normal, target_normal) < 0:
        normal = -normal

    # Rodrigues rotation formula
    v = np.cross(normal, target_normal)
    c_val = np.dot(normal, target_normal)
    I = np.eye(3)
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    if c_val > 0.99999:
        R = I
    elif c_val < -0.99999:
        R = -I
        R[0, 0] = 1
    else:
        s = 1 / (1 + c_val)
        R = I + vx + np.matmul(vx, vx) * s

    # Build transform: rotate then translate to put floor at Y=0
    rotated_centroid = np.dot(R, centroid)
    T = np.eye(4)
    T[:3, :3] = R
    T[1, 3] = -rotated_centroid[1]

    aligned_points = apply_transform_to_points(all_points, T)

    return aligned_points, T


# =============================================================================
# Point Cloud Filtering
# =============================================================================

def dbscan_filter(
    points: np.ndarray,
    eps: float = 0.05,
    min_samples: int = 32,
) -> np.ndarray:
    """
    Filter point cloud using DBSCAN clustering, keeping only clustered points.

    Removes noise/outlier points that don't belong to any dense cluster.

    Args:
        points: Input point cloud of shape (N, 3)
        eps: Maximum distance between neighbors in a cluster
        min_samples: Minimum points required to form a cluster

    Returns:
        Filtered point cloud with outliers removed
    """
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(points).labels_
    mask = labels != -1  # Keep non-noise points
    return points[mask]


# =============================================================================
# Visualization Utilities
# =============================================================================

def plot_dots(uv: np.ndarray, W: int, H: int) -> np.ndarray:
    """
    Rasterize 2D points into a grayscale image.

    Args:
        uv: 2D point coordinates of shape (N, 2)
        W: Image width
        H: Image height

    Returns:
        Grayscale image of shape (H, W) with points rasterized
    """
    img = np.zeros((H, W), dtype=np.float32)

    x = np.clip(uv[:, 0], 0, W - 1).astype(int)
    y = np.clip(uv[:, 1], 0, H - 1).astype(int)
    np.add.at(img, (y, x), 1)

    return (img * 255).astype(np.uint8)


def project_points_to_image(
    points_world: np.ndarray,
    camera_intrinsics: np.ndarray,
    cam2world: np.ndarray,
    W: int,
    H: int,
) -> np.ndarray:
    """
    Project 3D world points onto image plane and rasterize.

    Args:
        points_world: 3D points in world space, shape (N, 3)
        camera_intrinsics: 3x3 camera intrinsic matrix
        cam2world: 4x4 camera-to-world transform
        W: Image width
        H: Image height

    Returns:
        Rasterized image showing projected points
    """
    K = camera_intrinsics[:3, :3]
    world2cam = np.linalg.inv(cam2world)

    # Transform to camera space
    points_camera = (world2cam[:3, :3] @ points_world.T).T + world2cam[:3, 3]

    # Project to image plane
    points_image = (K @ points_camera.T).T
    points_image = points_image / points_image[:, 2:3]

    # Filter to valid image coordinates
    valid_mask = (
        (points_image[:, 0] > 0)
        & (points_image[:, 0] < W)
        & (points_image[:, 1] > 0)
        & (points_image[:, 1] < H)
    )
    points_image = points_image[valid_mask]

    return plot_dots(points_image[:, :2], W, H)


# =============================================================================
# Image Encoding
# =============================================================================

def jpg_encode(image: np.ndarray) -> bytes:
    """
    Encode numpy image array to JPEG bytes.

    Args:
        image: Image array of shape (H, W) or (H, W, 3)

    Returns:
        JPEG-encoded bytes
    """
    image = Image.fromarray(image)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    """
    Main processing pipeline: image -> depth -> point cloud -> ShapeR pickle.
    """
    # -------------------------------------------------------------------------
    # 1. Load model and run inference
    # -------------------------------------------------------------------------
    device = torch.device("cuda:0")
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
    model = model.to(device=device)

    image_paths = ["example/cup_painting.jpg"]
    prediction = model.inference(image=image_paths)

    target_size = (prediction.depth.shape[-1], prediction.depth.shape[-2])

    # ShapeR released checkpoint expects grayscale input
    image = np.array(
        Image.open("example/cup_painting.jpg")
        .resize(target_size, resample=Image.LANCZOS)
        .convert("L")
    )

    # -------------------------------------------------------------------------
    # 2. Generate raw point cloud from depth
    # -------------------------------------------------------------------------
    raw_metric_depth = merge_depth_maps_to_pointcloud(
        depths=prediction.depth,
        intrinsics=prediction.intrinsics,
        extrinsics=to_homogeneous_44(prediction.extrinsics),
        images=prediction.processed_images,
        conf=prediction.conf,
        conf_thresh=1.0,
        remove_bottom_percentile=5.0,
    )

    # -------------------------------------------------------------------------
    # 3. Load masks for foreground object and ground plane
    # -------------------------------------------------------------------------
    # Ideally: use instance detector + SAM2 to automatically segment objects
    # Here we use pre-computed interactive SAM2 masks as a workaround

    mask_fg = (
        np.array(
            Image.open("example/foreground.png").resize(target_size, resample=Image.NEAREST)
        )[None, ...]
        / 255
    )
    mask_xy = (
        np.array(
            Image.open("example/xy_plane.png").resize(target_size, resample=Image.NEAREST)
        )[None, ...]
        / 255
    )

    # -------------------------------------------------------------------------
    # 4. Extract foreground and ground plane points
    # -------------------------------------------------------------------------
    points_foreground = merge_depth_maps_to_pointcloud(
        depths=prediction.depth * mask_fg,
        intrinsics=prediction.intrinsics,
        extrinsics=to_homogeneous_44(prediction.extrinsics),
        images=prediction.processed_images,
        conf=prediction.conf,
        conf_thresh=1.0,
        remove_bottom_percentile=5.0,
    )

    # ShapeR expects Z-up coordinate system
    # With IMU you'd use gravity; here we estimate from a horizontal surface
    points_xyplane = merge_depth_maps_to_pointcloud(
        depths=prediction.depth * mask_xy,
        intrinsics=prediction.intrinsics,
        extrinsics=to_homogeneous_44(prediction.extrinsics),
        images=prediction.processed_images,
        conf=prediction.conf,
        conf_thresh=1.0,
        remove_bottom_percentile=5.0,
    )

    # -------------------------------------------------------------------------
    # 5. Align to ground plane and convert to Z-up
    # -------------------------------------------------------------------------
    caption = Path("example/caption.txt").read_text()

    # Align so ground plane becomes horizontal
    points_aligned, T_align = align_to_xy_plane(points_foreground, points_xyplane)

    # Convert from Y-down to Z-up coordinate system
    points_aligned_z_up = apply_transform_to_points(points_aligned, T_Y_DOWN_TO_Z_UP)

    # Center at origin
    _, T_center = center_box(points_aligned_z_up)
    points_aligned_z_up_center = apply_transform_to_points(points_aligned_z_up, T_center)

    # -------------------------------------------------------------------------
    # 6. Transform camera poses to match point cloud
    # -------------------------------------------------------------------------
    camera_poses = world2cam_to_cam2world(to_homogeneous_44(prediction.extrinsics))[0]
    camera_poses_aligned_z_up = T_Y_DOWN_TO_Z_UP @ T_align @ camera_poses
    camera_poses_aligned_z_up_center = T_center @ camera_poses_aligned_z_up

    # -------------------------------------------------------------------------
    # 7. Normalize scale and apply filtering
    # -------------------------------------------------------------------------
    half_size = (
        points_aligned_z_up_center.max(axis=0) - points_aligned_z_up_center.min(axis=0)
    ) / 2

    # Scale to fit in [-0.9, 0.9] box
    scale = 0.9 / np.max(half_size)
    points_aligned_z_up_center = points_aligned_z_up_center * scale

    # Remove outliers with DBSCAN
    points_aligned_z_up_center = dbscan_filter(
        points_aligned_z_up_center, eps=0.1, min_samples=8
    )

    # Downsample with FPS
    if points_aligned_z_up_center.shape[0] > 16:
        n_samples = min(1024, points_aligned_z_up_center.shape[0])
        fps_samples_idx = fpsample.fps_sampling(points_aligned_z_up_center, n_samples)
        points_aligned_z_up_center = points_aligned_z_up_center[fps_samples_idx, :]

    # Undo scale for final output (ShapeR handles normalization internally)
    points_aligned_z_up_center = points_aligned_z_up_center / scale

    # -------------------------------------------------------------------------
    # 8. Generate point cloud projection mask
    # -------------------------------------------------------------------------
    point_mask = project_points_to_image(
        points_aligned_z_up_center,
        prediction.intrinsics[0],
        camera_poses_aligned_z_up_center,
        mask_fg.shape[-1],
        mask_fg.shape[-2],
    )

    # -------------------------------------------------------------------------
    # 9. Build and save ShapeR pickle
    # -------------------------------------------------------------------------
    # Convert arrays to tensors
    points_tensor = torch.from_numpy(points_aligned_z_up_center).float()
    camera_tensor = torch.from_numpy(camera_poses_aligned_z_up_center).float()
    half_size_tensor = torch.from_numpy(half_size).float()
    intrinsics_tensor = torch.from_numpy(prediction.intrinsics[0])
    raw_depth_tensor = torch.from_numpy(raw_metric_depth)
    T_model_world = torch.from_numpy(T_center @ T_Y_DOWN_TO_Z_UP @ T_align)

    pkl_sample = {
        # Point cloud and geometry
        "points_model": points_tensor,
        "bounds": half_size_tensor,
        "T_model_world": T_model_world,

        # Uncertainty estimates (zeros for single-view)
        "inv_dist_std": torch.zeros_like(points_tensor)[:, 0],
        "dist_std": torch.zeros_like(points_tensor)[:, 0],

        # Image and camera data
        "image_data": [jpg_encode(image)],
        "camera_to_worlds": [camera_tensor],
        "camera_params": [intrinsics_tensor],
        "mask_data": [jpg_encode(point_mask)],

        # Metadata
        "caption": caption,
        "experimental_dav3": True,
        "raw_depth": raw_depth_tensor,
    }

    output_path = "example/cup_painting.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(pkl_sample, f)

    print(f"Saved {output_path}")
    print("Now you can infer this with ShapeR with:")
    print("python infer_shape.py --input_pkl experimental/example/cup_painting.pkl --config balance --is_local_path")
    print("to get the prediction centered at origin and aligned to xy plane")
    print("OR")
    print("python infer_shape.py --input_pkl experimental/example/cup_painting.pkl --config balance --is_local_path --do_transform_to_world")
    print("to get the shape reconstructed in the same frame as raw_depth extracted from DepthAnythingv3")


if __name__ == "__main__":
    main()
