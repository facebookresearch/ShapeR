# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
ShapeR Denoiser: Flow matching model for 3D shape generation.

Generates latent codes conditioned on point clouds, images, and text.
Uses a dual-stream transformer architecture with multi-modal conditioning.
"""

# pyre-unsafe

import copy
import random

import torch
import torch.nn as nn

from model.dino_and_ray_feature_extractor import DinoAndRayFeatureExtractor
from model.flow_matching.dualstream_transformer import (
    FlowMatchingTransformer,
    MLPEmbedder,
)
from model.flow_matching.helpers.model_wrapper import ModelWrapper
from model.flow_matching.helpers.scheduler import FluxTimeSampler
from model.flow_matching.helpers.solver import ODESolver

from model.pointcloud_encoder import PointCloudEncoder
from model.text.hf_embedder import DummyTextFeatureExtractor
from torchsparse import SparseTensor
from tqdm import tqdm

dummy_text_extractor = DummyTextFeatureExtractor(device=None)


class ShapeRDenoiser(nn.Module):
    """
    Flow matching denoiser for 3D shape reconstruction.

    Conditions on: point clouds (sparse conv), images (DINO), text (T5/CLIP).
    Generates VAE latent codes that decode to 3D meshes.
    """

    def __init__(self, config):
        super().__init__()
        self.input_types = config.encoder.type
        self.x0_mode = config.encoder.x0_mode
        self.simple_latent_projection = MLPEmbedder(
            in_dim=config.vae.embed_dim, hidden_dim=config.encoder.d_model
        )
        self.variable_num_views = config.dataset.variable_num_views
        self.use_pre_text_attn_blocks = config.encoder.use_pre_text_attn_blocks
        feat_dim = 768
        if "image" in self.input_types:
            self.dino_ray_extractor = DinoAndRayFeatureExtractor(
                patchified_grid_size=config.encoder.dino_image_size // 14,
                post_inject_masks=config.encoder.dino_mask_inject,
                rot90inputs=config.encoder.dino_rot90inputs,
                large_splat_size=-1,
                use_giant_model=config.encoder.dino_use_giant_model,
                use_upsampler=config.encoder.dino_legacy_upsample,
                use_sam2_features=getattr(config.encoder, "use_sam2_features", False),
            )
            self.image_cond_separator = torch.nn.Parameter(
                torch.rand(1, 1, self.dino_ray_extractor.get_feat_dims()) * 1e-2,
                requires_grad=True,
            )
            feat_dim = self.dino_ray_extractor.get_feat_dims()

        if "point" in self.input_types:
            self.pc_num_bins = config.encoder.num_bins
            self.point_cloud_feature_extractor = PointCloudEncoder(
                input_channels=config.encoder.input_channels,
                d_model=feat_dim,
                conv_layers=config.encoder.conv_layers,
                num_bins=config.encoder.num_bins,
            )
            self.point_cond_separator = torch.nn.Parameter(
                torch.rand(1, 1, feat_dim) * 1e-2,
                requires_grad=True,
            )

        if "text" in self.input_types:
            self.text_cond_separator = torch.nn.Parameter(
                torch.rand(1, 1, feat_dim) * 1e-2,
                requires_grad=True,
            )
            # assume t5-xl
            self.simple_t5_projection = MLPEmbedder(in_dim=2048, hidden_dim=feat_dim)
            # assume clip-vit-large-patch14
            self.simple_clip_projection = MLPEmbedder(in_dim=768, hidden_dim=256)

        self.null_context = torch.nn.Parameter(
            torch.zeros(1, 1, feat_dim),
            requires_grad=False,
        )

        self.transformer = FlowMatchingTransformer(
            in_channels=config.encoder.d_model,
            out_channels=config.vae.embed_dim,
            use_context_in="image" in self.input_types,
            context_in_dim=feat_dim,
            use_txt_in="text" in self.input_types,
            vec_in_dim=256,
            use_pre_text_attn=self.use_pre_text_attn_blocks,
            config=config.fm_transformer,
        )

    def convert_to_bfloat16(self):
        self.transformer = self.transformer.to(torch.bfloat16)
        self.simple_latent_projection = self.simple_latent_projection.to(torch.bfloat16)
        # keep the point cloud feature extractor in float32
        self.point_cloud_feature_extractor = self.point_cloud_feature_extractor.to(
            torch.float32
        )
        if "image" in self.input_types:
            self.dino_ray_extractor = self.dino_ray_extractor.to(torch.bfloat16)
            self.image_cond_separator = torch.nn.Parameter(
                self.image_cond_separator.to(torch.bfloat16)
            )
        if "point" in self.input_types:
            self.point_cond_separator = torch.nn.Parameter(
                self.point_cond_separator.to(torch.bfloat16)
            )
        if "text" in self.input_types:
            self.simple_t5_projection = self.simple_t5_projection.to(torch.bfloat16)
            self.simple_clip_projection = self.simple_clip_projection.to(torch.bfloat16)
            self.text_cond_separator = torch.nn.Parameter(
                self.text_cond_separator.to(torch.bfloat16)
            )
        self.null_context = torch.nn.Parameter(self.null_context.to(torch.bfloat16))

    # @property
    # def dtype(self):
    #     next(self.parameters()).dtype

    def forward(
        self, x_t, t, batch, unconditional="all_condition", return_intermediate=False
    ):
        if unconditional == "all_condition" or unconditional == "one_condition":
            if "point" in self.input_types:
                pc_cond = batch["semi_dense_points"]
                # pc_cond.feats = pc_cond.feats.to(x_t.dtype)
            else:
                pc_cond = None

            if "image" in self.input_types:
                img_in = batch["images"]
                cam_ext = batch["camera_extrinsics"]
                cam_int = batch["camera_intrinsics"]
                msk_in = batch["masks_ingest"]
                box_in = batch["boxes_ingest"]
            else:
                img_in = None
                cam_ext = None
                cam_int = None
                msk_in = None

            txt_in_t5 = batch["t5_text"]
            txt_in_clip = batch["clip_text"]

            skipped_condition = None
            if unconditional == "one_condition":
                unskipped_condition = random.choices(
                    ["point", "image", "text"], [0.2, 0.35, 0.45], k=1
                )[0]
                skipped_condition = [
                    cond
                    for cond in self.input_types.split("_")
                    if cond != unskipped_condition
                ]

            return self.forward_impl(
                x_t,
                pc_cond,
                img_in,
                msk_in,
                box_in,
                cam_ext,
                cam_int,
                txt_in_t5,
                txt_in_clip,
                t,
                skipped_condition,
                return_intermediate=return_intermediate,
            )
        elif unconditional == "no_condition":
            return self.forward_uncond_impl(
                x_t, t, return_intermediate=return_intermediate
            )

    def get_x0_from_input(self, batch, token_shape=None):
        x_0 = None
        x0_mode = self.x0_mode
        if token_shape is not None:
            B, L, F = token_shape
        else:
            B, L, F = batch["z_0"].shape
        if x0_mode == "random":
            x_0 = torch.randn(
                (B, L, F),
                device=batch["semi_dense_points"].feats.device,
            )
        else:
            raise NotImplementedError(f"Unknown x0_mode: {x0_mode}")
        # dtype should match the model dtype
        x_0 = x_0.to(self.null_context.dtype)
        return x_0

    def forward_uncond_impl(self, z_x_t, t, return_intermediate=False):
        num_in_tokens = z_x_t.shape[1]
        z_x_t_in = self.simple_latent_projection(z_x_t)

        condition_context = []
        txt_tokens = None
        if "point" in self.input_types:
            point_tokens = self.null_context.expand(z_x_t_in.shape[0], -1, -1)
            condition_context.append(point_tokens)
            condition_context.append(
                self.point_cond_separator.expand(z_x_t_in.shape[0], -1, -1)
            )
        if "image" in self.input_types:
            img_tokens = self.null_context.expand(z_x_t_in.shape[0], -1, -1)
            condition_context.append(img_tokens)
            condition_context.append(
                self.image_cond_separator.expand(z_x_t_in.shape[0], -1, -1)
            )
        if "text" in self.input_types:
            txt_tokens = self.null_context.expand(z_x_t_in.shape[0], -1, -1)
            condition_context.append(txt_tokens)
            condition_context.append(
                self.text_cond_separator.expand(z_x_t_in.shape[0], -1, -1)
            )

        if len(condition_context) > 0:
            condition_context = torch.cat(condition_context, dim=1)
        else:
            condition_context = None

        output = self.transformer(
            z_x_t_in, condition_context, t, y=None, txt_tokens=txt_tokens
        )
        output, intermediate = output
        output = output[:, :num_in_tokens, :]
        if return_intermediate:
            return output, intermediate
        return output

    def forward_impl(
        self,
        z_x_t,
        pc_cond,
        img_in,
        msk_in,
        box_in,
        cam_ext,
        cam_int,
        txt_in_t5,
        txt_in_clip,
        t,
        skipped_condition,
        return_intermediate=False,
    ):
        z_x_t_in = self.simple_latent_projection(z_x_t)

        condition_context = []
        clip_tokens = None
        if "point" in self.input_types:
            # if True:
            if skipped_condition is not None and "point" in skipped_condition:
                point_tokens = self.null_context.expand(z_x_t_in.shape[0], -1, -1)
                condition_context.append(point_tokens)
                condition_context.append(
                    self.point_cond_separator.expand(z_x_t_in.shape[0], -1, -1)
                )
            else:
                pc_features = self.point_cloud_feature_extractor(pc_cond)
                condition_context.append(pc_features["context"].to(z_x_t_in.dtype))
                condition_context.append(
                    self.point_cond_separator.expand(
                        condition_context[-1].shape[0], -1, -1
                    )
                )

        if "image" in self.input_types:
            # if True:
            if skipped_condition is not None and "image" in skipped_condition:
                img_tokens = self.null_context.expand(z_x_t_in.shape[0], -1, -1)
                condition_context.append(img_tokens)
                condition_context.append(
                    self.image_cond_separator.expand(z_x_t_in.shape[0], -1, -1)
                )
            else:
                B, V, C, H, W = img_in.shape
                sequence_idx = torch.arange(B).to(img_in.device)
                sequence_idx = sequence_idx.unsqueeze(1).repeat(1, V).reshape(-1)
                img_in = img_in.reshape(B * V, C, H, W)
                msk_in = msk_in.reshape(B * V, 1, H, W)
                box_in = box_in.reshape(B * V, 2, 2)
                camera_extrinsics = cam_ext.reshape(B * V, 4, 4)
                camera_intrinsics = cam_int.reshape(B * V, 4, 4)
                dino_feats = self.dino_ray_extractor(
                    img_in,
                    camera_extrinsics,
                    camera_intrinsics,
                    msk_in,
                    box_in,
                )
                img_tokens = self.dino_ray_extractor.get_token_feature(dino_feats)
                if self.training and self.variable_num_views:
                    img_tokens = img_tokens.reshape(
                        B, V, img_tokens.shape[1], img_tokens.shape[2]
                    )
                    selected_img_tokens = []
                    for v_idx in range(V):
                        v_toks = sorted(
                            random.sample(list(range(img_tokens.shape[2])), 1024 // V)
                        )
                        selected_img_tokens.append(
                            img_tokens[:, v_idx : v_idx + 1, v_toks, :]
                        )
                    img_tokens = torch.cat(selected_img_tokens, dim=1)
                img_tokens = img_tokens.reshape(B, -1, img_tokens.shape[-1])
                condition_context.append(img_tokens)
                condition_context.append(
                    self.image_cond_separator.expand(
                        condition_context[-1].shape[0], -1, -1
                    )
                )

        txt_tokens = None
        if "text" in self.input_types:
            if skipped_condition is not None and "text" in skipped_condition:
                txt_tokens = self.null_context.expand(z_x_t_in.shape[0], -1, -1)
                condition_context.append(txt_tokens)
                condition_context.append(
                    self.text_cond_separator.expand(z_x_t_in.shape[0], -1, -1)
                )
            else:
                txt_tokens = self.simple_t5_projection(txt_in_t5)
                if not self.use_pre_text_attn_blocks:
                    condition_context.append(txt_tokens)
                else:
                    condition_context.append(
                        self.null_context.expand(z_x_t_in.shape[0], -1, -1)
                    )
                condition_context.append(
                    self.text_cond_separator.expand(
                        condition_context[-1].shape[0], -1, -1
                    )
                )
                clip_tokens = self.simple_clip_projection(txt_in_clip)

        if len(condition_context) > 0:
            condition_context = torch.cat(condition_context, dim=1)
        else:
            condition_context = None

        output = self.transformer(
            z_x_t_in, condition_context, t, y=clip_tokens, txt_tokens=txt_tokens
        )
        output, intermediate = output
        if return_intermediate:
            return output, intermediate
        return output

    def infer_latents_simple_euler(
        self,
        batch,
        token_shape,
        prior_sampler,
        force_fixed_num_images=None,
        text_feature_extractor=dummy_text_extractor,
        num_steps=20,
    ):
        x_t = prior_sampler(self, batch, token_shape)
        b = x_t.shape[0]
        velocity_model = WrappedModel(
            self, batch, text_feature_extractor, force_fixed_num_images
        )
        for i in tqdm(range(num_steps), desc="infer latents"):
            t = torch.full((b,), i / num_steps, device=x_t.device)[0]
            t_next = torch.full((b,), (i + 1) / num_steps, device=x_t.device)[0]
            dt = t_next - t
            vel_pred = velocity_model(x=x_t, t=t)
            x_t = x_t + dt * vel_pred
        return x_t

    def infer_latents(
        self,
        batch,
        token_shape,
        force_fixed_num_images=None,
        text_feature_extractor=dummy_text_extractor,
        num_steps=10,
        cfg_value=-1,
        use_shifted_sampling=False,
    ):
        """
        Generate latent codes from conditioning inputs using ODE solver.

        Args:
            batch: Dict with point cloud, images, text, camera params
            token_shape: (batch, num_tokens, embed_dim) for output latents
            num_steps: Number of ODE solver steps
            cfg_value: Classifier-free guidance scale (-1 = disabled)
            use_shifted_sampling: Use Flux-style time shifting

        Returns:
            Latent codes of shape token_shape
        """
        step_size = None
        # step_size = 0.01
        if not use_shifted_sampling:
            T = torch.linspace(0, 1, num_steps)  # sample times
            T = T.to(device=batch["semi_dense_points"].feats.device)
        else:
            T = FluxTimeSampler(mode="inference")(
                num_steps,
                # token_shape[0],
                min(token_shape[0], 2048 * 2),  # only allow max 4096 token shifting
                device=batch["semi_dense_points"].feats.device,
            )
        solver = ODESolver(
            velocity_model=WrappedModel(
                self,
                batch,
                text_feature_extractor,
                force_fixed_num_images,
                cfg_value=cfg_value,
            )
        )
        x_init = self.get_x0_from_input(batch, token_shape=token_shape)
        sol = solver.sample(
            time_grid=T,
            x_init=x_init,
            method="midpoint",
            step_size=step_size,
            return_intermediates=False,
        )  # sample from the model
        return sol

    def infer_latents_sequence(
        self,
        batch,
        force_fixed_num_images=None,
        text_feature_extractor=dummy_text_extractor,
    ):
        step_size = 0.025
        T = torch.linspace(0, 1, 25)  # sample times
        T = T.to(device=batch["z_0"].device)
        solver = ODESolver(
            velocity_model=WrappedModel(
                self, batch, text_feature_extractor, force_fixed_num_images
            )
        )
        x_init = self.get_x0_from_input(batch)
        sol = solver.sample(
            time_grid=T,
            x_init=x_init,
            method="midpoint",
            step_size=step_size,
            return_intermediates=True,
        )  # sample from the model
        return sol.permute(1, 0, 2, 3)


class WrappedModel(ModelWrapper):
    """Wraps ShapeRDenoiser for use with ODE solver, handling batch preparation."""

    def __init__(
        self,
        model: nn.Module,
        batch: dict,
        text_feature_extractor=dummy_text_extractor,
        force_fixed_num_images=None,
        cfg_value=-1,
    ):
        super().__init__(model)
        self.model = model
        self.batch = copy.deepcopy(batch)
        self.text_feature_extractor = text_feature_extractor
        if force_fixed_num_images is not None:
            self.batch["images"] = self.batch["images"][:, :force_fixed_num_images]
            self.batch["masks_ingest"] = self.batch["masks_ingest"][
                :, :force_fixed_num_images
            ]
            self.batch["boxes_ingest"] = self.batch["boxes_ingest"][
                :, :force_fixed_num_images
            ]
            self.batch["camera_extrinsics"] = self.batch["camera_extrinsics"][
                :, :force_fixed_num_images
            ]
            self.batch["camera_intrinsics"] = self.batch["camera_intrinsics"][
                :, :force_fixed_num_images
            ]
        self.batch["t5_text"], self.batch["clip_text"] = text_feature_extractor(
            batch["caption"]
        )
        self.cfg_value = cfg_value
        if cfg_value != -1:
            self.forward = self.forward_CFG
        else:
            self.forward = self.forward_no_CFG

    def forward_no_CFG(self, x: torch.Tensor, t: torch.Tensor, **extras):
        pc_cond = self.batch["semi_dense_points"]
        self.batch["semi_dense_points"] = SparseTensor(
            pc_cond.feats, pc_cond.coords, pc_cond.stride
        )
        t = t[None].expand(x.shape[0])
        return self.model(x, t, self.batch)

    def forward_CFG(self, x: torch.Tensor, t: torch.Tensor, **extras):
        pc_cond = self.batch["semi_dense_points"]
        self.batch["semi_dense_points"] = SparseTensor(
            pc_cond.feats, pc_cond.coords, pc_cond.stride
        )
        t = t[None].expand(x.shape[0])
        condition_v = self.model(x, t, self.batch)
        no_condition_v = self.model(x, t, self.batch, unconditional="no_condition")
        out = no_condition_v + self.cfg_value * (condition_v - no_condition_v)
        return out
