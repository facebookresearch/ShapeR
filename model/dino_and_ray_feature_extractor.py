# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from dataclasses import dataclass
from types import MethodType
from typing import Any, Dict

import torch
from model.dinov2.hub.backbones import dinov2_vitl14_reg, dinov2_vits14_reg
from model.structure.utils import (
    UpsampleX2Conv2dResBlock,
    UpsampleX4Conv2dResBlock,
    UpsampleX8Conv2dResBlock,
)
from model.unet import MaskDownsamplingNet

from preprocessing.ray_utils import get_image_ray_plucker
from torchvision import transforms

DINO_FEATURE_NAME = "dino"
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


@dataclass(eq=False)
class DinoAndRayFeatureExtractor(torch.nn.Module):
    # Note, we use the plucker coordinate conditioning from
    # https://openreview.net/attachment?id=H4yQefeXhp&name=pdf
    backbone_arch: str = "vits14_reg"
    freeze: bool = True
    patchified_grid_size: int = 32  # patch embed produces [30 x 30] grid
    pre_inject_plucker: bool = False
    post_inject_plucker_dim: int = 64
    post_inject_mask_dim: int = 64
    post_inject_sam2_dim: int = 64
    post_inject_masks: bool = False
    separate_dino_lr: bool = False
    add_cls_token: bool = True
    delete_dino_mask_token: bool = False
    large_splat_size: int = -1
    rot90inputs: bool = False
    use_giant_model: bool = False
    use_upsampler: bool = True
    use_sam2_features: bool = False

    @property
    def returns_grid_features(self) -> bool:
        return False

    def convert_to_bfloat16(self):
        self._dino = self._dino.to(torch.bfloat16)

    def __post_init__(self):
        super().__init__()
        torch.hub.set_dir("/tmp/dinov2")
        if self.use_giant_model:
            backbone_model = dinov2_vitl14_reg(pretrained=True)
            self.post_inject_mask_dim = 256
            self.post_inject_plucker_dim = 256
            if self.use_sam2_features:
                self.post_inject_plucker_dim = 320
                self.post_inject_mask_dim = 128
                self.post_inject_sam2_dim = 256
        else:
            backbone_model = dinov2_vits14_reg(pretrained=True)
            self.post_inject_mask_dim = 64
            self.post_inject_plucker_dim = 64
            if self.use_sam2_features:
                self.post_inject_plucker_dim = 80
                self.post_inject_mask_dim = 32
                self.post_inject_sam2_dim = 64

        assert self.patchified_grid_size in [
            16,
            20,
            32,
            37,
        ], "Only 16, 20, 32 and 37 are supported at the moment."

        # Adjust the prepare_tokens_with_masks method of dino to also take as input the
        # plucker ray coordinates.
        # If self.pre_inject_plucker==True, we add the plucker patch embedding
        # to the image patch embedding, the plucker patch embedding is 0-initialized
        # If self.pre_inject_plucker==False, the plucker coordinates are added to the
        # extracted dino features
        if self.freeze:
            backbone_model = backbone_model.eval()

        self._image_size = [
            self.patchified_grid_size * s for s in backbone_model.patch_embed.patch_size
        ]

        backbone_model = self._set_dino_prepare_tokens_with_masks(backbone_model)
        self._dino = backbone_model

        assert backbone_model.patch_embed.patch_size == (14, 14)

        # the raysampler outputs ray origins and directions for dino
        self._feat_dim = {}
        self._feat_dim[DINO_FEATURE_NAME] = self._dino.embed_dim

        self._aux_unet_upsampler = None
        if self.use_upsampler:
            self._upsample_image_features = (
                torch.nn.Identity()
                if self.patchified_grid_size == 32
                else UpsampleX2Conv2dResBlock(self.get_feat_dims())
            )
        if self.large_splat_size == 256:
            self._aux_unet_upsampler = UpsampleX8Conv2dResBlock(self.get_feat_dims())
        elif self.large_splat_size == 128:
            self._aux_unet_upsampler = UpsampleX4Conv2dResBlock(self.get_feat_dims())
        elif self.large_splat_size == 64:
            self._aux_unet_upsampler = UpsampleX2Conv2dResBlock(self.get_feat_dims())
        elif self.large_splat_size != -1:
            raise NotImplementedError("Only 128 is supported at the moment.")

        if self.freeze:
            assert (
                not self.pre_inject_plucker
            ), "Cannot train the pre-inject plucker embedder when the trunk is frozen."
            freeze_all_params(self._dino, self.freeze)
            if self.use_sam2_features:
                freeze_all_params(self.sam2_feature_extractor, self.freeze)
        elif self.delete_dino_mask_token:
            # This parameter is unused during forward and raises error in DDP
            del self._dino.mask_token

    def _dino_img_transform(self, img):
        n = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        return n(img)

    def _set_dino_prepare_tokens_with_masks(self, backbone_model):
        patch_emb = backbone_model.patch_embed

        def _make_patch_embed(in_chans):
            l = type(patch_emb)(
                img_size=patch_emb.img_size,
                patch_size=patch_emb.patch_size,
                in_chans=in_chans,
                embed_dim=(
                    patch_emb.embed_dim
                    if self.pre_inject_plucker
                    else self.post_inject_plucker_dim
                ),
            )
            # init the patch embed with very small weights to not destroy the
            # pretrained image features
            l.proj.weight.data.normal_(0.0, 1e-2)
            l.proj.bias.data.zero_()
            return l

        self._plucker_patch_embed = _make_patch_embed(6)
        if self.post_inject_masks:
            self._mask_patch_embed = MaskDownsamplingNet(
                img_size=self._image_size,
                patch_size=patch_emb.patch_size,
                in_chans=1,
                embed_dim=(self.post_inject_mask_dim),
            )

        if self.pre_inject_plucker:

            def prepare_tokens_with_masks_and_plucker(self_dino, x, masks=None):
                x, rest = x[:, :3], x[:, 3:]

                B, nc, w, h = x.shape
                x = self_dino.patch_embed(x)

                # add the plucker patch embedding to the image patch embedding
                plucker, rest = rest[:, :6], rest[:, 6:]
                plucker_emb = self._plucker_patch_embed(plucker)
                x = x + plucker_emb

                # continue as before
                if masks is not None:
                    x = torch.where(
                        masks.unsqueeze(-1),
                        self_dino.mask_token.to(x.dtype).unsqueeze(0),
                        x,
                    )
                x = torch.cat(
                    (self_dino.cls_token.expand(x.shape[0], -1, -1), x),
                    dim=1,
                )
                x = x + self_dino.interpolate_pos_encoding(x, w, h)
                if self_dino.register_tokens is not None:
                    x = torch.cat(
                        (
                            x[:, :1],
                            self_dino.register_tokens.expand(x.shape[0], -1, -1),
                            x[:, 1:],
                        ),
                        dim=1,
                    )

                return x

            backbone_model.prepare_tokens_with_masks = MethodType(
                prepare_tokens_with_masks_and_plucker,
                backbone_model,
            )

        if not self.pre_inject_plucker:
            # post inject plucker
            self._plucker_patch_embed = torch.nn.Sequential(
                self._plucker_patch_embed,
                torch.nn.LayerNorm(self.post_inject_plucker_dim, eps=1e-6),
            )

        if self.post_inject_masks:
            self._mask_patch_embed = torch.nn.Sequential(
                self._mask_patch_embed,
                torch.nn.LayerNorm(self.post_inject_mask_dim, eps=1e-6),
            )

        return backbone_model

    def get_feat_dims(self) -> int:
        # pyre-fixme[29]
        return sum(self._feat_dim.values()) + (
            self.post_inject_plucker_dim * int(not self.pre_inject_plucker)
            + self.post_inject_mask_dim * int(self.post_inject_masks)
            + self.post_inject_sam2_dim * int(self.use_sam2_features)
        )

    def get_token_feature(self, feats: Dict[Any, torch.Tensor]) -> torch.Tensor:
        return torch.cat(list(feats.values()), dim=-1)

    def get_image_grid_feature(
        self, feats: Dict[Any, torch.Tensor], aux_full_res_feats=None
    ) -> torch.Tensor:
        # feats in a single image reshaped to the original size
        H = W = self.patchified_grid_size
        grid_feature = feats[DINO_FEATURE_NAME]
        assert grid_feature.shape[1] == H * W + 1
        grid_image_feature = grid_feature[:, : (H * W)].reshape(
            grid_feature.shape[0],
            H,
            W,
            -1,
        )
        # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
        grid_image_feature = self._upsample_image_features(grid_image_feature)
        if self._aux_unet_upsampler is not None:
            if self.large_splat_size in [64, 128, 256]:  # non unet style -- only dino
                # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
                grid_image_feature = self._aux_unet_upsampler(grid_image_feature)
            else:
                raise NotImplementedError("Only 64/128 is supported at the moment.")

        return grid_image_feature

    def forward(
        self, imgs, camera_to_world, camera_intrinsics, masks=None, boxes=None
    ) -> Dict[Any, torch.Tensor]:
        # interpolate to the DINO size
        # imgs = torch.nn.functional.interpolate(imgs, size=self._image_size)
        imgs_original = imgs.clone()
        imgs = self._dino_img_transform(imgs)

        # extract the rays + their plucker
        B, _, H, W = imgs.shape

        plucker = get_image_ray_plucker(
            camera_to_world,
            camera_intrinsics,
            H,
            W,
        ).permute(0, 3, 1, 2)

        dino_input = imgs

        if self.pre_inject_plucker:
            # run dino on the concatenation of imgs and plucker
            dino_input = torch.cat([dino_input, plucker], dim=1)

        if self.rot90inputs:
            dino_input = torch.rot90(dino_input, 3, [2, 3])
        # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
        #  `forward_features`.
        dino_out = self._dino.forward_features(dino_input, masks=None)
        if self.rot90inputs:
            H_p = W_p = self.patchified_grid_size
            B_p, HW_p, C_p = dino_out["x_norm_patchtokens"].shape
            assert (
                HW_p == H_p * W_p
            ), f"DINO feature shape mismatch. {H_p}x{W_p} != {HW_p}"
            image_features = dino_out["x_norm_patchtokens"].reshape(
                B_p,
                H_p,
                W_p,
                -1,
            )
            image_features = torch.rot90(image_features, 1, [1, 2])
            dino_out["x_norm_patchtokens"] = image_features.reshape(
                B_p,
                -1,
                C_p,
            )

        dino_feats = dino_out["x_norm_patchtokens"]
        cls_token = dino_out["x_norm_clstoken"]

        assert (
            dino_feats.shape[1] == self.patchified_grid_size**2
        ), "DINO feature shape mismatch."

        if self.freeze:
            dino_feats = dino_feats.detach()
            cls_token = cls_token.detach()

        if not self.pre_inject_plucker:
            # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
            plucker_feats = self._plucker_patch_embed(plucker)
            aux_feats = [plucker_feats]
            if self.post_inject_masks:
                # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
                mask_feats = self._mask_patch_embed(masks)
                aux_feats.append(mask_feats)
            if self.use_sam2_features:
                sam2_feats = self.sam2_feature_extractor(imgs_original, boxes)
                sam2_feats = self.sam2_feature_postprocess(sam2_feats)
                aux_feats.append(sam2_feats)
            dino_feats = torch.cat([dino_feats] + aux_feats, dim=-1)
            # we need to pad the cls token to allow appending
            cls_token = torch.nn.functional.pad(
                cls_token,
                (0, sum([f.shape[-1] for f in aux_feats])),
            )

        dino_and_cls = (
            torch.cat([dino_feats, cls_token[:, None]], dim=1)
            if self.add_cls_token
            else dino_feats
        )

        out_feats = {}
        out_feats[DINO_FEATURE_NAME] = dino_and_cls
        return out_feats


def freeze_all_params(
    module_or_parameter,
    freeze: bool = True,
) -> None:
    if isinstance(module_or_parameter, torch.nn.Parameter):
        module_or_parameter.requires_grad = not freeze
        return
    for p in module_or_parameter.parameters():
        p.requires_grad = not freeze
