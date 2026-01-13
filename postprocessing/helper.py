# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Postprocessing utilities for ShapeR outputs.

Includes mesh cleanup (floating geometry removal) and visualization functions
for comparing predictions with ground truth.
"""

import trimesh


def remove_floating_geometry(mesh):
    """Remove small disconnected components, keeping those with >=5% of largest component's faces."""
    # Decompose the mesh into connected components
    components = mesh.split(only_watertight=False)
    # Find the largest component by comparing the number of faces
    largest_component = max(components, key=lambda c: c.faces.shape[0])
    faces_ratio = [
        c.faces.shape[0] / largest_component.faces.shape[0] for c in components
    ]
    qualifying_components = [
        c for i, c in enumerate(components) if faces_ratio[i] >= 0.05
    ]
    # merge the qualifying components
    largest_component = trimesh.util.concatenate(qualifying_components)
    return largest_component


def render_mesh_to_image(mesh, color, resolution=(512, 512)):
    """
    Render a trimesh mesh to an image using pyrender.

    Assumes mesh is centered at origin with coordinates in [-0.9, 0.9] range.
    Uses Z-up viewing angle for consistent visualization with point cloud.
    Works on headless machines without a display.
    """
    import os

    # Set offscreen rendering backend before importing pyrender
    # Try EGL first (faster on GPU), fall back to OSMesa (CPU software rendering)
    if "PYOPENGL_PLATFORM" not in os.environ:
        os.environ["PYOPENGL_PLATFORM"] = "egl"

    try:
        import pyrender
    except Exception:
        # If EGL fails, try OSMesa
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        import pyrender

    import numpy as np

    # Create a copy and apply color
    mesh_copy = mesh.copy()
    mesh_copy.visual.vertex_colors = np.array(color + [255], dtype=np.uint8)

    # Convert to pyrender mesh
    py_mesh = pyrender.Mesh.from_trimesh(mesh_copy)

    # Create scene
    scene = pyrender.Scene(bg_color=[255, 255, 255, 255])
    scene.add(py_mesh)

    # Camera setup: Z-up view, looking from front-right-top angle
    # Matches the point cloud visualization viewpoint
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)  # Narrower FOV for larger appearance
    camera_pose = np.eye(4)
    # Position camera to view Z-up mesh from elevated front-right angle
    # elev=25, azim=45 in matplotlib corresponds to this camera position
    camera_pose[:3, 3] = np.array([2.0, 2.0, 1.5])
    # Look at origin with Z as up vector
    camera_pose[:3, :3] = _look_at_rotation(
        eye=camera_pose[:3, 3],
        target=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, 0.0, 1.0]),  # Z-up
    )
    scene.add(camera, pose=camera_pose)

    # Add lights from multiple directions for better visibility
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=camera_pose)

    # Add fill light from opposite side
    fill_light_pose = np.eye(4)
    fill_light_pose[:3, 3] = np.array([-2.0, -1.0, 1.0])
    fill_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.5)
    scene.add(fill_light, pose=fill_light_pose)

    # Render
    renderer = pyrender.OffscreenRenderer(*resolution)
    color_img, _ = renderer.render(scene)
    renderer.delete()

    return color_img


def _look_at_rotation(eye, target, up):
    """Compute rotation matrix for camera looking at target from eye position."""
    import numpy as np

    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    up_new = np.cross(right, forward)

    rotation = np.eye(3)
    rotation[0, :] = right
    rotation[1, :] = up_new
    rotation[2, :] = -forward

    return rotation.T


def render_pointcloud_to_image(points, resolution=(512, 512)):
    """
    Render a point cloud to an image using matplotlib 3D projection.

    Assumes point cloud is centered at origin in [-0.9, 0.9] coordinate system.
    Uses Z-up viewing angle matching mesh rendering.
    Works on headless machines without a display.
    """
    import io

    import matplotlib

    matplotlib.use("Agg")  # Use non-GUI backend for headless environments
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(6, 6), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    # Subsample if too many points
    if len(points) > 5000:
        indices = np.random.choice(len(points), 5000, replace=False)
        points = points[indices]

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, c="#088F8F", alpha=0.7)  # Blue Green

    # Fixed axis limits for origin-centered data in [-0.9, 0.9] range
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)

    # Remove axis labels and ticks for cleaner look
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")

    # Set Z-up view matching the mesh rendering (elev=25, azim=45)
    ax.view_init(elev=25, azim=45)

    # Make the plot fill more of the figure
    ax.set_box_aspect([1, 1, 1])

    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1, facecolor="white")
    buf.seek(0)
    plt.close(fig)

    from PIL import Image

    img = Image.open(buf)
    img = img.resize(resolution)
    return np.array(img)


def visualize_prediction_and_groundtruth(
    predicted_mesh,
    target_mesh,
    point_cloud,
    input_images,
    masks,
    caption,
    sample_name,
    save_path=None,
    figsize=(20, 8),
):
    """
    Create a compact visualization combining all inputs and outputs.

    Layout: [Images/Masks/Caption] | [Point Cloud] | [Predicted] | [Target]

    Args:
        predicted_mesh: trimesh.Trimesh - the predicted mesh
        target_mesh: trimesh.Trimesh - the ground truth mesh
        point_cloud: np.ndarray (N, 3) - input point cloud
        input_images: list of np.ndarray or PIL.Image - N input images (typically 16)
        masks: list of np.ndarray or PIL.Image - N 2D point image masks
        caption: str - text caption describing the shape
        sample_name: str - name of the sample (used as title)
        save_path: str - optional path to save the figure
        figsize: tuple - figure size

    Returns:
        fig: matplotlib figure object
    """
    import matplotlib

    matplotlib.use("Agg")  # Use non-GUI backend for headless environments
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    # Convert images to numpy if needed
    def to_numpy(img):
        if isinstance(img, Image.Image):
            return np.array(img)
        if isinstance(img, np.ndarray):
            if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
                # CHW format, convert to HWC
                img = np.transpose(img, (1, 2, 0))
            if img.dtype == np.float32 or img.dtype == np.float64:
                img = (img * 255).astype(np.uint8)
            return img
        return img

    input_images = [to_numpy(img) for img in input_images]
    masks = [to_numpy(m) for m in masks]

    # Helper to wrap caption text at N words per line
    def wrap_text(text, words_per_line=10):
        words = text.split()
        lines = []
        for i in range(0, len(words), words_per_line):
            lines.append(" ".join(words[i:i + words_per_line]))
        return "\n".join(lines)

    n_images = len(input_images)
    n_cols_img = 4 if n_images <= 4 else 8  # Use 4 columns for small sets, 8 for larger
    n_rows_img = (n_images + n_cols_img - 1) // n_cols_img

    # Stack images into grids
    img_h, img_w = input_images[0].shape[:2]

    # Create image grid
    img_grid = np.ones((n_rows_img * img_h, n_cols_img * img_w, 3), dtype=np.uint8) * 255
    for i, img in enumerate(input_images):
        row = i // n_cols_img
        col = i % n_cols_img
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 4:
            img = img[..., :3]
        img_grid[row * img_h:(row + 1) * img_h, col * img_w:(col + 1) * img_w] = img

    # Create mask grid
    mask_grid = np.ones((n_rows_img * img_h, n_cols_img * img_w, 3), dtype=np.uint8) * 255
    for i, mask in enumerate(masks):
        row = i // n_cols_img
        col = i % n_cols_img
        if mask.ndim == 2:
            mask = np.stack([mask] * 3, axis=-1)
        elif mask.shape[-1] == 4:
            mask = mask[..., :3]
        mask_grid[row * img_h:(row + 1) * img_h, col * img_w:(col + 1) * img_w] = mask

    # Calculate total left column height (images + masks + caption space + labels + padding)
    top_padding = int(img_h * 0.08)  # Small top padding
    label_height = int(img_h * 0.25)  # Height for section labels (increased for more gap)
    bottom_padding = int(img_h * 0.15)  # Bottom padding
    # Calculate caption lines needed
    wrapped_caption = wrap_text(caption, words_per_line=10)
    n_caption_lines = len(wrapped_caption.split("\n"))
    caption_height = int(img_h * (0.35 + 0.12 * n_caption_lines))  # Dynamic height based on lines
    left_total_height = top_padding + label_height + img_grid.shape[0] + label_height + mask_grid.shape[0] + caption_height + bottom_padding

    # Render 3D visualizations at resolution matching the left column height
    render_size = left_total_height
    pc_image = render_pointcloud_to_image(point_cloud, resolution=(render_size, render_size))
    pred_image = render_mesh_to_image(predicted_mesh, color=[255, 192, 0], resolution=(render_size, render_size))  # Golden Yellow #FFC000
    target_image = render_mesh_to_image(target_mesh, color=[34, 139, 34], resolution=(render_size, render_size))  # Forest Green #228B22

    # Create figure using subplots with width ratios
    # Left: images/masks/caption, Right: 3 equal-width 3D views
    fig, axes = plt.subplots(
        1, 4,
        figsize=figsize,
        facecolor="white",
        gridspec_kw={
            "width_ratios": [n_cols_img * img_w, render_size, render_size, render_size],
            "wspace": 0.02,
        },
    )

    # === Left Column: Stack images, masks, and caption vertically ===
    # Create a combined image for the left column
    left_width = n_cols_img * img_w
    left_combined = np.ones((left_total_height, left_width, 3), dtype=np.uint8) * 255

    # Track vertical position
    y_offset = 0

    # Top padding
    y_offset += top_padding

    # Leave space for "Input Images" label (will be added via text)
    y_offset += label_height

    # Place image grid
    left_combined[y_offset:y_offset + img_grid.shape[0], :] = img_grid
    y_offset += img_grid.shape[0]

    # Leave space for "Masks" label
    y_offset += label_height

    # Place mask grid
    left_combined[y_offset:y_offset + mask_grid.shape[0], :] = mask_grid

    axes[0].imshow(left_combined)
    axes[0].axis("off")

    # Add section labels using axes coordinates
    # "Input Images" label at top (after top_padding)
    input_label_y = 1.0 - (top_padding + label_height * 0.5) / left_total_height
    axes[0].text(
        0.5,
        input_label_y,
        "Input Images",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        transform=axes[0].transAxes,
    )

    # "Masks" label between images and masks
    masks_label_y = 1.0 - (top_padding + label_height + img_grid.shape[0] + label_height * 0.5) / left_total_height
    axes[0].text(
        0.5,
        masks_label_y,
        "Masks",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        transform=axes[0].transAxes,
    )

    # "Caption" label and caption text at bottom
    caption_label_y = (bottom_padding + caption_height * 0.85) / left_total_height
    axes[0].text(
        0.5,
        caption_label_y,
        "Caption",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        transform=axes[0].transAxes,
    )

    caption_text_y = (bottom_padding + caption_height * 0.3) / left_total_height
    axes[0].text(
        0.5,
        caption_text_y,
        wrapped_caption,
        ha="center",
        va="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
        transform=axes[0].transAxes,
    )

    # === Right Columns: 3D Visualizations ===
    axes[1].imshow(pc_image)
    axes[1].axis("off")
    axes[1].set_title("Point Cloud", fontsize=11, fontweight="bold", pad=8)

    axes[2].imshow(pred_image)
    axes[2].axis("off")
    axes[2].set_title("Predicted", fontsize=11, fontweight="bold", pad=8)

    axes[3].imshow(target_image)
    axes[3].axis("off")
    axes[3].set_title("Ground Truth", fontsize=11, fontweight="bold", pad=8)

    # Use subplots_adjust for tight layout
    plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02, wspace=0.02)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.02)
        print(f"Visualization saved to {save_path}")

    return fig
