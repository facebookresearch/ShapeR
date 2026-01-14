# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal, Optional, Tuple, Union

import einops

import numpy as np

import torch

from .pose import IdentityPose, PoseTW
from .projection_utils import (
    fisheye624_project,
    fisheye624_unproject,
    pinhole_project,
    pinhole_unproject,
)
from .tensor_wrapper import autocast, autoinit, smart_cat, TensorWrapper

RGB_PARAMS = np.float32(
    # pyre-fixme[6]: For 1st argument expected `Union[None, bytes, str,
    #  SupportsFloat, SupportsIndex]` but got `List[float]`.
    [2 * 600.0, 2 * 352.0, 2 * 352.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
)
# pyre-fixme[6]: For 1st argument expected `Union[None, bytes, str, SupportsFloat,
#  SupportsIndex]` but got `List[float]`.
SLAM_PARAMS = np.float32([500.0, 320.0, 240.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

FISHEYE624_TYPE_STR = (
    "FisheyeRadTanThinPrism:f,u0,v0,k0,k1,k2,k3,k5,k5,p1,p2,s1,s2,s3,s4"
)
FISHEYE624_DF_TYPE_STR = (
    "FisheyeRadTanThinPrism:fu,fv,u0,v0,k0,k1,k2,k3,k5,k5,p1,p2,s1,s2,s3,s4"
)
PINHOLE_TYPE_STR = "Pinhole"


def get_T_rot_z(angle: float):
    T_rot_z = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0, 0.0],
            [np.sin(angle), np.cos(angle), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    return torch.from_numpy(T_rot_z).float()


def is_fisheye624(inp):
    names = [
        "Fisheye624",
        "f624",
        FISHEYE624_TYPE_STR,
        FISHEYE624_DF_TYPE_STR,
        "FisheyeRadTanThinPrism",
    ]
    names += [name.lower() for name in names]
    return inp in names


def is_kb3(inp):
    names = ["KB:fu,fv,u0,v0,k0,k1,k2,k3", "KannalaBrandtK3", "KB3"]
    names += [name.lower() for name in names]
    return inp in names


def is_pinhole(inp):
    names = ["Pinhole", "Linear"]
    names += [name.lower() for name in names]
    return inp in names


class DefaultCameraTWParam(TensorWrapper):
    """Allows multiple input sizes."""

    def __init__(self):
        self._data = -1 * torch.ones(15)

    @property
    def shape(self):
        return (torch.Size([16]), torch.Size([15]), torch.Size([8]), torch.Size([4]))


class DefaultCameraTWDistParam(TensorWrapper):
    """Allows multiple input sizes."""

    def __init__(self):
        self._data = -1 * torch.ones(12)

    @property
    def shape(self):
        return (torch.Size([12]), torch.Size([4]), torch.Size([0]))


DEFAULT_CAM_PARAM = DefaultCameraTWParam()
DEFAULT_CAM_DIST_PARAM = DefaultCameraTWDistParam()
DEFAULT_CAM_DATA_SIZE = 34
DEFAULT_ARIAGEN2_RGB_W = 2016
DEFAULT_ARIAGEN2_RGB_H = 1512
DEFAULT_ARIAGEN2_SLAM_H = 512
DEFAULT_ARIAGEN2_SLAM_W = 512


class CameraTW(TensorWrapper):
    """
    Class to represent a batch of camera calibrations of the same camera type.
    """

    SIZE_IND = slice(0, 2)
    F_IND = slice(2, 4)
    C_IND = slice(4, 6)
    GAIN_IND = 6
    EXPOSURE_S_IND = 7
    VALID_RADIUS_IND = slice(8, 10)
    T_CAM_RIG_IND = slice(10, 22)
    DIST_IND = slice(22, None)
    _valid_dims = (22, 26, 34)

    @autocast
    @autoinit
    def __init__(self, data: Optional[torch.Tensor] = None):
        if data is None:
            data = -1 * torch.ones(34)
        assert isinstance(data, torch.Tensor)
        assert data.shape[-1] in [22, 26, 34], f"Invalid shape {data.shape[-1]}"
        super().__init__(data)

    # TODO: Use None as default values, using mutable tensors is risky.
    @classmethod
    @autoinit
    def from_parameters(
        cls,
        width: torch.Tensor = -1 * torch.ones(1),
        height: torch.Tensor = -1 * torch.ones(1),
        fx: torch.Tensor = -1 * torch.ones(1),
        fy: torch.Tensor = -1 * torch.ones(1),
        cx: torch.Tensor = -1 * torch.ones(1),
        cy: torch.Tensor = -1 * torch.ones(1),
        gain: torch.Tensor = -1 * torch.ones(1),
        exposure_s: torch.Tensor = 1e-3 * torch.ones(1),
        valid_radiusx: torch.Tensor = 99999.0 * torch.ones(1),
        valid_radiusy: torch.Tensor = 99999.0 * torch.ones(1),
        T_camera_rig: Union[torch.Tensor, PoseTW] = IdentityPose,
        dist_params: Union[
            torch.Tensor, DefaultCameraTWDistParam
        ] = DEFAULT_CAM_DIST_PARAM,
    ):
        # Concatenate into one big data tensor, handles TensorWrapper objects.
        data = smart_cat(
            [
                width,
                height,
                fx,
                fy,
                cx,
                cy,
                gain,
                exposure_s,
                valid_radiusx,
                valid_radiusy,
                T_camera_rig,
                dist_params,
            ],
            dim=-1,
        )
        return cls(data)

    @classmethod
    @autoinit
    def from_surreal(
        cls,
        width: torch.Tensor = -1 * torch.ones(1),
        height: torch.Tensor = -1 * torch.ones(1),
        type_str: str = "Fisheye624",
        params: Union[torch.Tensor, DefaultCameraTWParam] = DEFAULT_CAM_PARAM,
        gain: torch.Tensor = 1 * torch.ones(1),
        exposure_s: torch.Tensor = 1e-3 * torch.ones(1),
        valid_radius: torch.Tensor = 99999.0 * torch.ones(1),
        T_camera_rig: Union[torch.Tensor, PoseTW] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        ),
    ):
        # Try to auto-determine the camera model.
        if (
            is_fisheye624(type_str) and params.shape[-1] == 16
        ):  # Fisheye624 double focals
            fx = params[..., 0].unsqueeze(-1)
            fy = params[..., 1].unsqueeze(-1)
            cx = params[..., 2].unsqueeze(-1)
            cy = params[..., 3].unsqueeze(-1)
            dist_params = params[..., 4:]
        elif (
            is_fisheye624(type_str) and params.shape[-1] == 15
        ):  # Fisheye624 single focal
            f = params[..., 0].unsqueeze(-1)
            cx = params[..., 1].unsqueeze(-1)
            cy = params[..., 2].unsqueeze(-1)
            dist_params = params[..., 3:]
            fx = fy = f
        elif is_kb3(type_str) and params.shape[-1] == 8:  # KB3.
            fx = params[..., 0].unsqueeze(-1)
            fy = params[..., 1].unsqueeze(-1)
            cx = params[..., 2].unsqueeze(-1)
            cy = params[..., 3].unsqueeze(-1)
            dist_params = params[..., 4:]
        elif is_pinhole(type_str) and params.shape[-1] == 4:  # Pinhole.
            fx = params[..., 0].unsqueeze(-1)
            fy = params[..., 1].unsqueeze(-1)
            cx = params[..., 2].unsqueeze(-1)
            cy = params[..., 3].unsqueeze(-1)
            dist_params = params[..., 4:]
        else:
            raise NotImplementedError(
                "Unknown number of params entered for camera model"
            )

        if torch.any(torch.logical_or(valid_radius > height, valid_radius > width)):
            if not is_pinhole(type_str):
                # Try to auto-determine the valid radius for fisheye cameras.
                default_radius = 99999.0
                hw_ratio = height / width
                eyevideo_camera_hw_ratio = torch.tensor(240.0 / 640.0).to(hw_ratio)
                slam_camera_hw_ratio = torch.tensor(480.0 / 640.0).to(hw_ratio)
                ariagen2_rgb_camera_hw_ratio = torch.tensor(
                    DEFAULT_ARIAGEN2_RGB_H / DEFAULT_ARIAGEN2_RGB_W
                ).to(hw_ratio)
                rgb_camera_hw_ratio = torch.tensor(2880.0 / 2880.0).to(hw_ratio)
                ariagen2_hw_match = torch.logical_and(
                    height == DEFAULT_ARIAGEN2_RGB_H, width == DEFAULT_ARIAGEN2_RGB_W
                )
                guess_rgb = hw_ratio == rgb_camera_hw_ratio
                guess_slam = hw_ratio == slam_camera_hw_ratio
                guess_ariagen2 = hw_ratio == ariagen2_rgb_camera_hw_ratio
                guess_slam = torch.logical_and(guess_slam, ~ariagen2_hw_match)
                guess_ariagen2 = torch.logical_and(guess_ariagen2, ariagen2_hw_match)
                guess_eyevideo = hw_ratio == eyevideo_camera_hw_ratio
                valid_radius = default_radius * torch.ones_like(hw_ratio)
                # Assume "Rogallo"/"IMX577" aka Aria RGB Camera.
                valid_radius = torch.where(
                    guess_rgb, 1415 * (height / 2880), valid_radius
                )
                # Assume "Canyon"/"OV7251" aka Aria SLAM camera.
                valid_radius = torch.where(
                    guess_slam, 330 * (height / 480), valid_radius
                )
                # Assume Aria Gen2 RGB camera.
                # 2x valid radius since its wrt to unbinned sensor size
                valid_radius = torch.where(
                    guess_ariagen2,
                    2 * 1512 * (height / DEFAULT_ARIAGEN2_RGB_H),
                    valid_radius,
                )
                # This is for Eye Video Camera
                valid_radius = torch.where(
                    guess_eyevideo, 330 * (height / 480), valid_radius
                )
                if torch.any(valid_radius == default_radius):
                    raise ValueError(
                        f"Failed to auto-determine valid radius based on aspect ratios (valid_radius {valid_radius}, width {width}, height {height})"
                    )
            else:
                # Note that the valid_radius for pinhole camera is not well-defined.
                # We heuristically set the valid radius to be the half of the image diagonal.
                # Add one pixel to be sure that all pixels in the image are valid.
                valid_radius = (
                    # pyre-fixme[58]: `**` is not supported for operand types
                    #  `Tensor` and `int`.
                    torch.sqrt((width / 2.0) ** 2 + (height / 2.0) ** 2) + 1.0
                )

        return cls.from_parameters(
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            gain=gain,
            exposure_s=exposure_s,
            valid_radiusx=valid_radius,
            valid_radiusy=valid_radius,
            T_camera_rig=T_camera_rig,
            dist_params=dist_params,
        )

    @property
    def size(self) -> torch.Tensor:
        """Size (width height) of the images, with shape (..., 2)."""
        return self._data[..., self.SIZE_IND]

    @property
    def f(self) -> torch.Tensor:
        """Focal lengths (fx, fy) with shape (..., 2)."""
        return self._data[..., self.F_IND]

    @property
    def c(self) -> torch.Tensor:
        """Principal points (cx, cy) with shape (..., 2)."""
        return self._data[..., self.C_IND]

    @property
    def K(self) -> torch.Tensor:
        """Intrinsic matrix with shape (..., 3, 3)"""
        K = torch.eye(3, device=self.device, dtype=self.dtype)
        # Make proper size of K to take care of B and T dims.
        K_view = [1] * (self.f.ndim - 1) + [3, 3]
        K_repeat = list(self.f.shape[:-1]) + [1, 1]
        K = K.view(K_view)
        K = K.repeat(K_repeat)
        K[..., 0, 0] = self.f[..., 0]
        K[..., 1, 1] = self.f[..., 1]
        K[..., 0, 2] = self.c[..., 0]
        K[..., 1, 2] = self.c[..., 1]
        return K

    @property
    def K44(self) -> torch.Tensor:
        """Intrinsic matrix with shape (..., 4, 4)"""
        K = torch.eye(4, device=self.device, dtype=self.dtype)
        # Make proper size of K to take care of B and T dims.
        K_view = [1] * (self.f.ndim - 1) + [4, 4]
        K_repeat = list(self.f.shape[:-1]) + [1, 1]
        K = K.view(K_view)
        K = K.repeat(K_repeat)
        K[..., 0, 0] = self.f[..., 0]
        K[..., 1, 1] = self.f[..., 1]
        K[..., 0, 2] = self.c[..., 0]
        K[..., 1, 2] = self.c[..., 1]
        return K

    @property
    def gain(self) -> torch.Tensor:
        """Gain of the camera, with shape (..., 1)."""
        return self._data[..., self.GAIN_IND].unsqueeze(-1)

    @property
    def exposure_s(self) -> torch.Tensor:
        """Exposure of the camera in seconds, with shape (..., 1)."""
        return self._data[..., self.EXPOSURE_S_IND].unsqueeze(-1)

    @property
    def valid_radius(self) -> torch.Tensor:
        """Radius from camera center for valid projections, with shape (..., 1)."""
        return self._data[..., self.VALID_RADIUS_IND]

    @property
    def T_camera_rig(self) -> PoseTW:
        """Pose of camera, shape (..., 12)."""
        return PoseTW(self._data[..., self.T_CAM_RIG_IND])

    @property
    def dist(self) -> torch.Tensor:
        """Distortion parameters, with shape (..., {0, D}), where D is number of distortion params."""
        return self._data[..., self.DIST_IND]

    @property
    def params(self) -> torch.Tensor:
        """Get the camera "params", which are defined as fx,fy,cx,cy,dist"""
        return torch.cat([self.f, self.c, self.dist], dim=-1)

    @property
    def is_fisheye624(self):
        return self.dist.shape[-1] == 12

    @property
    def is_kb3(self):
        return self.dist.shape[-1] == 4

    @property
    def is_linear(self):
        return self.dist.shape[-1] == 0

    def set_valid_radius(self, valid_radius: torch.Tensor):
        self._data[..., self.VALID_RADIUS_IND] = valid_radius

    def set_T_camera_rig(self, T_camera_rig: PoseTW):
        self._data[..., self.T_CAM_RIG_IND] = T_camera_rig._data.clone()

    def set_f(self, f) -> torch.Tensor:
        self._data[..., self.F_IND] = f

    def set_size(self, size) -> torch.Tensor:
        self._data[..., self.SIZE_IND] = size

    def scale(self, scales: Union[float, int, Tuple[Union[float, int]]]):
        """Update the camera parameters after resizing an image."""
        if isinstance(scales, (int, float)):
            scales = (scales, scales)
        s = self._data.new_tensor(scales)
        data = torch.cat(
            [
                self.size * s,
                self.f * s,
                (self.c + 0.5) * s - 0.5,
                self.gain,
                self.exposure_s,
                self.valid_radius * s,
                # pyre-fixme[16]: `Tensor` has no attribute `_data`.
                self.T_camera_rig._data,
                self.dist,
            ],
            dim=-1,
        )
        return self.__class__(data)

    def scale_to_size(self, size_wh: Union[int, Tuple[int]]):
        """Scale the camera parameters to a given image size"""
        if torch.unique(self.size).numel() > 2:
            raise ValueError(f"cannot handle multiple sizes {self.size}")
        if isinstance(size_wh, int):
            size_wh = (size_wh, size_wh)
        i0w = tuple([0] * self.ndim)
        i0h = tuple([0] * (self.ndim - 1) + [1])
        scale = (
            float(size_wh[0]) / float(self.size[i0w]),
            float(size_wh[1]) / float(self.size[i0h]),
        )
        # pyre-fixme[6]: For 1st argument expected `Union[Tuple[Union[float, int]],
        #  float, int]` but got `Tuple[float, float]`.
        return self.scale(scale)

    def scale_to(self, im: torch.Tensor):
        """
        Scale the camera parameters to match the size of the given image assumes
        ...xHxW image tensor convention of pytorch
        """
        H, W = im.shape[-2:]
        # pyre-fixme[6]: For 1st argument expected `Union[Tuple[int], int]` but got
        #  `Tuple[int, int]`.
        return self.scale_to_size((W, H))

    def crop(self, left_top: Tuple[float], size: Tuple[int]):
        """Update the camera parameters after cropping an image."""
        left_top = self._data.new_tensor(left_top)
        size = self._data.new_tensor(size)

        # Expand the dimension if self._data is a tensor of CameraTW
        if len(self._data.shape) > 1:
            expand_dim = list(self._data.shape[:-1]) + [1]
            size = size.repeat(expand_dim)
            left_top = left_top.repeat(expand_dim)

        data = torch.cat(
            [
                size,
                self.f,
                self.c - left_top,
                self.gain,
                self.exposure_s,
                self.valid_radius,
                # pyre-fixme[16]: `Tensor` has no attribute `_data`.
                self.T_camera_rig._data,
                self.dist,
            ],
            dim=-1,
        )
        return self.__class__(data)

    @autocast
    def in_image(self, p2d: torch.Tensor):
        """Check if 2D points are within the image boundaries."""
        assert p2d.shape[-1] == 2, f"p2d shape needs to be 2d {p2d.shape}"
        # assert p2d.shape[:-2] == self.shape  # allow broadcasting
        size = self.size.unsqueeze(-2)
        valid = torch.all((p2d >= 0) & (p2d <= (size - 1)), dim=-1)
        return valid

    @autocast
    def in_radius(self, p2d: torch.Tensor):
        """Check if 2D points are within the valid fisheye radius region."""
        assert p2d.shape[-1] == 2, f"p2d shape needs to be 2d {p2d.shape}"
        dists = torch.linalg.norm(
            (p2d - self.c.unsqueeze(-2)) / self.valid_radius.unsqueeze(-2),
            dim=-1,
            ord=2,
        )
        valid = dists < 1.0
        return valid

    @autocast
    def in_radius_mask(self):
        """
        Return a mask that is True where 2D points are within the valid fisheye
        radius region.  Returned mask is of shape ... x 1 x H x W, where ... is
        the shape of the camera (BxT or B for example).
        """
        s = self.shape[:-1]
        C = self.shape[-1]
        px = pixel_grid(self.view(-1, C)[0])
        H, W, _ = px.shape
        valids = self.in_radius(px.view(-1, 2))
        s = s + (1, H, W)
        valids = valids.view(s)
        return valids

    @autocast
    def in_fov(self, p3d: torch.Tensor, fov_deg: float):
        """Check if 3D points are within the valid FOV of the fisheye camera."""
        assert p3d.shape[-1] == 3, f"p3d shape needs to be 3d {p3d.shape}"
        # unproject the principal point to get the principal axis
        principal, _ = self.unproject(self.c.view(-1, 1, 2))
        principal = principal / torch.norm(principal, dim=-1, keepdim=True)
        # dot(principal, p3d) / (norm(principal) * norm(p3d))
        cos_angle = torch.sum(principal * p3d, dim=-1) / torch.norm(p3d, dim=-1)
        rad = torch.acos(cos_angle)
        fov_rad = np.deg2rad(fov_deg)
        valid = torch.isfinite(rad) & (rad < (fov_rad / 2.0)) & (cos_angle >= 0)
        return valid

    @autocast
    def project(
        self, p3d: torch.Tensor, cam: Optional[Literal["rgb", "slaml", "slamr"]] = None
    ) -> Tuple[torch.Tensor]:
        """Transform 3D points into 2D pixel coordinates.
        cam can be provided to enable additional FoV check"""

        # Explicitly promote the data types.
        promoted_type = torch.promote_types(self._data.dtype, p3d.dtype)
        self._data = self._data.to(promoted_type)
        p3d = p3d.to(promoted_type)

        # Try to auto-determine the camera model.
        if self.is_fisheye624:  # Fisheye624.
            params = torch.cat([self.f, self.c, self.dist], dim=-1)
            if params.ndim == 1:
                B = p3d.shape[0]
                params = params.unsqueeze(0).repeat(B, 1)
            p2d = fisheye624_project(p3d, params)
        elif self.is_linear:  # Pinhole.
            params = self.params
            if params.ndim == 1:
                B = p3d.shape[0]
                params = params.unsqueeze(0).repeat(B, 1)
            p2d = pinhole_project(p3d, params)
        else:
            raise ValueError(
                "only fisheye624 and pinhole implemented, kb3 not yet implemented"
            )

        in_image = self.in_image(p2d)
        in_radius = self.in_radius(p2d)
        in_front = p3d[..., -1] > 0
        valid = in_image & in_radius & in_front

        if cam is not None:
            # FOV values (in degrees) from Project Aria paper
            fov_dict = {
                "rgb": 110.0,
                "slaml": 120.0,
                "slamr": 120.0,
            }
            if cam not in fov_dict:
                raise ValueError("fov only available for rgb/slaml/slamr")
            fov_deg = fov_dict[cam]
            valid = valid & self.in_fov(p3d, fov_deg)

        # pyre-fixme[7]: Expected `Tuple[Tensor]` but got `Tuple[Any, Any]`.
        #  Expected has length 1, but actual has length 2.
        return p2d, valid

    @autocast
    def unproject(self, p2d: torch.Tensor) -> Tuple[torch.Tensor]:
        """Transform 2D points into 3D rays."""

        # Explicitly promote the data types.
        promoted_type = torch.promote_types(self._data.dtype, p2d.dtype)
        self._data = self._data.to(promoted_type)
        p2d = p2d.to(promoted_type)

        # Try to auto-determine the camera model.
        if self.is_fisheye624:  # Fisheye624.
            params = torch.cat([self.f, self.c, self.dist], dim=-1)
            if params.ndim == 1:
                B = p2d.shape[0]
                params = params.unsqueeze(0).repeat(B, 1)
            rays = fisheye624_unproject(p2d, params)
        elif self.is_linear:  # Pinhole.
            params = self.params
            if params.ndim == 1:
                B = p2d.shape[0]
                params = params.unsqueeze(0).repeat(B, 1)
            rays = pinhole_unproject(p2d, params)
        else:
            raise ValueError(
                "only fisheye624 and pinhole implemented, kb3 not yet implemented"
            )

        in_image = self.in_image(p2d)
        in_radius = self.in_radius(p2d)
        valid = in_image & in_radius
        # pyre-fixme[7]: Expected `Tuple[Tensor]` but got `Tuple[Any, Any]`.
        #  Expected has length 1, but actual has length 2.
        return rays, valid

    def rotate_90_cw(self):
        return self.rotate_90(clock_wise=True)

    def rotate_90_ccw(self):
        return self.rotate_90(clock_wise=False)

    def rotate_90(self, clock_wise: bool):
        dist_params = self.dist.clone()
        if self.is_fisheye624:
            # swap thin prism and tangential distortion parameters
            # {k_0 ... k_5} {p_0 p_1} {s_0 s_1 s_2 s_3} to
            # {k_0 ... k_5} {p_1 p_0} {s_2 s_3 s_0 s_1}
            dist_p = self.dist[..., 6:8]
            dist_s = self.dist[..., 8:12]
            dist_params[..., 6] = dist_p[..., 1]
            dist_params[..., 7] = dist_p[..., 0]
            dist_params[..., 8:10] = dist_s[..., 2:]
            dist_params[..., 10:12] = dist_s[..., :2]
        elif self.is_linear:
            # no need to rotate distortion parameters since there are none
            pass
        elif self.is_kb3:
            raise NotImplementedError(f"kb3 model rotation not implemented yet")
        else:
            raise NotImplementedError(f"camera model not recognized {self}")

        # clock-wise or counter clock-wise
        DIR = 1 if clock_wise else -1
        # rotate camera extrinsics by 90 degree CW
        T_rot_z = PoseTW.from_matrix3x4(get_T_rot_z(DIR * np.pi * 0.5)).to(self.device)
        if clock_wise:
            # rotate x, y of principal point
            # x_rotated = height - 1 - y_before
            # y_rotated = x_before
            rot_cx = self.size[..., 1] - self.c[..., 1] - 1
            rot_cy = self.c[..., 0].clone()
        else:
            rot_cx = self.c[..., 1].clone()
            rot_cy = self.size[..., 0] - self.c[..., 0] - 1

        return CameraTW.from_parameters(
            # swap width and height
            self.size[..., 1].clone().unsqueeze(-1),
            self.size[..., 0].clone().unsqueeze(-1),
            # swap x, y of focal lengths
            self.f[..., 1].clone().unsqueeze(-1),
            self.f[..., 0].clone().unsqueeze(-1),
            rot_cx.unsqueeze(-1),
            rot_cy.unsqueeze(-1),
            self.gain.clone(),
            self.exposure_s.clone(),
            # swap valid radius x, y
            self.valid_radius[..., 1].clone().unsqueeze(-1),
            self.valid_radius[..., 0].clone().unsqueeze(-1),
            # rotate camera extrinsics
            T_rot_z @ self.T_camera_rig,
            dist_params,
        )

    def __repr__(self):
        return f"CameraTW {self.shape} {self.dtype} {self.device}"


def grid_2d(
    width: int,
    height: int,
    output_range=(-1.0, 1.0, -1.0, 1.0),
    device="cpu",
    dtype=torch.float32,
):
    x = torch.linspace(
        output_range[0], output_range[1], width + 1, device=device, dtype=dtype
    )[:-1]
    y = torch.linspace(
        output_range[2], output_range[3], height + 1, device=device, dtype=dtype
    )[:-1]
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    grid = torch.stack([xx, yy], dim=-1)
    return grid


def pixel_grid(cam: CameraTW):
    assert cam.ndim == 1, f"Camera must be 1 dimensional {cam.shape}"
    W, H = int(cam.size[0]), int(cam.size[1])
    return grid_2d(W, H, output_range=[0, W, 0, H], device=cam.device, dtype=cam.dtype)


def rectify_video(
    video_snippet: torch.Tensor,
    fisheye_cam: CameraTW,
    pinhole_fxy_factor: Optional[float] = None,
    interp_mode: str = "bicubic",
    padding_mode: str = "zeros",
    # pyre-fixme[31]: Expression `(torch.Tensor,
    #  surreal.spaceport.utils.camera.CameraTW)` is not a valid type.
) -> (torch.Tensor, CameraTW):
    non_batch = False
    if video_snippet.ndim == 4:
        video_snippet = video_snippet.unsqueeze(0)  # add batch dim
        fisheye_cam = fisheye_cam.unsqueeze(0)
        non_batch = True

    assert video_snippet.ndim == 5 and (
        video_snippet.shape[-3] == 3 or video_snippet.shape[-3] == 1
    ), f"{video_snippet.shape}"
    # The values are chosen to retain the FoV as much as possible while minimizing blackout region, see D54285700.
    if not pinhole_fxy_factor:
        if video_snippet.shape[-3] == 3:  # rgb
            pinhole_fxy_factor = 1.0
        else:  # slam
            pinhole_fxy_factor = 1.11

    has_T = video_snippet.ndim == 5
    fx = fisheye_cam.f[..., 0:1].clone() / pinhole_fxy_factor
    fy = fisheye_cam.f[..., 1:2].clone() / pinhole_fxy_factor
    # centralize principle points.
    cx = (fisheye_cam.size.clone()[..., 0:1] - 1.0) / 2.0
    cy = (fisheye_cam.size.clone()[..., 1:2] - 1.0) / 2.0
    f_scaled = torch.cat([fx, fy], -1)
    c_center = torch.cat([cx, cy], -1)
    width = fisheye_cam.size[..., 0:1]
    height = fisheye_cam.size[..., 1:2]
    pinhole_cam = CameraTW.from_surreal(
        width=width,
        height=height,
        type_str="pinhole",
        params=torch.cat([f_scaled, c_center], -1),
        gain=fisheye_cam.gain,
        exposure_s=fisheye_cam.exposure_s,
        # will be set to diagonal automatically inside constructor
        valid_radius=fisheye_cam.valid_radius[..., 0] * 10,
        T_camera_rig=fisheye_cam.T_camera_rig,
    )
    H, W = int(height.view(-1)[0].item()), int(width.view(-1)[0].item())

    video_snippet, pinhole_cam = source_to_target(
        video_snippet,
        fisheye_cam,
        pinhole_cam,
        H,
        W,
        has_T,
        non_batch,
        interp_mode,
        padding_mode,
    )
    return video_snippet, pinhole_cam


def source_to_target(
    source_snippet,
    source_cam,
    target_cam,
    H,
    W,
    has_T,
    non_batch,
    interp_mode,
    padding_mode,
):
    """
    Given a snippet of source images, source cameras and target cameras,
    return the images of the target snippet and the target cameras.
    """
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W))
    yy, xx = yy.to(source_cam.device), xx.to(source_cam.device)
    target = torch.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)]).unsqueeze(0)
    if has_T:
        target = target.expand(source_snippet.shape[1], *target.shape[1:]).unsqueeze(0)
    target = target.expand(source_snippet.shape[0], *target.shape[1:])
    rays, valid = target_cam.unproject(target)
    # Note: one could buffer the `source` as LookUpTables in some way to speed up.
    source, valid = source_cam.project(rays)
    s = *source_snippet.shape[:2], H, W, 2
    if not has_T:
        s = *source_snippet.shape[:1], H, W, 2
    source = source.reshape(*s)
    width_ori = source_cam.size[..., 0:1].clone()
    height_ori = source_cam.size[..., 1:2].clone()
    H_ori, W_ori = int(height_ori.view(-1)[0].item()), int(width_ori.view(-1)[0].item())
    source[..., 0] = 2 * (source[..., 0] / W_ori) - 1
    source[..., 1] = 2 * (source[..., 1] / H_ori) - 1
    T = None
    if has_T:
        T = source_snippet.shape[1]
        # reshape source
        source_snippet = einops.rearrange(source_snippet, "b t c h w -> (b t) c h w")
        source = einops.rearrange(source, "b t h w c -> (b t) h w c")
        # reshape img
    target_snippet = torch.nn.functional.grid_sample(
        source_snippet.clone().detach(),
        source,
        interp_mode,
        padding_mode=padding_mode,
        align_corners=False,
    )
    if has_T:
        target_snippet = einops.rearrange(
            target_snippet, "(b t) c h w -> b t c h w", t=T
        )

    if non_batch:
        target_snippet = target_snippet.squeeze(0)
        target_cam = target_cam.squeeze(0)
    return target_snippet, target_cam


def param_to_matrix(params):
    fx, fy, cx, cy = params
    K = torch.tensor(
        [
            [fx, 0, cx, 0],
            [0, fy, cy, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        device=params.device,
    )
    return K
