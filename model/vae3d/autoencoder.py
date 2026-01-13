# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.
#
# This file is inspired by the DORA-VAE project.
# Original source: https://github.com/Seed3D/Dora
# Original license: Apache License 2.0

"""
3D VAE for mesh generation from latent codes.

Perceiver-style encoder/decoder with cross-attention for encoding point clouds
and decoding to occupancy/UDF values for mesh extraction.
"""

import math
import os

import numpy as np
import omegaconf
import torch
import torch.nn as nn

from einops import repeat
from model.vae3d.attention import ResidualCrossAttentionBlock, SelfAttentionTransformer
from model.vae3d.utils import (
    AutoEncoder,
    DiagonalGaussianDistribution,
    FourierEmbedder,
    get_embedder,
)

from preprocessing.helper import get_parameters_from_state_dict

from torch_cluster import fps


class SharpCoarseCrossAttentionEncoder(nn.Module):
    """Perceiver encoder: compresses coarse+sharp point clouds to latent tokens via cross-attention."""

    def __init__(
        self,
        use_downsample: bool,
        num_latents: int,
        embedder: FourierEmbedder,
        point_feats: int,
        width: int,
        heads: int,
        layers: int,
        token_scales: list,
        token_probability: list,
        init_scale: float,
        qkv_bias: bool,
        use_ln_post: bool,
    ):
        super().__init__()
        self.num_latents = num_latents
        self.use_downsample = use_downsample
        self.tokens = np.array(token_scales)
        self.train_probabilities = np.array(token_probability)
        self.infer_probabilities = np.zeros_like(self.train_probabilities)
        self.infer_probabilities[-2] = 1.0
        if not self.use_downsample:
            self.query = nn.Parameter(torch.randn((num_latents, width)) * 0.02)

        self.embedder = embedder

        self.input_proj = nn.Linear(self.embedder.out_dim + point_feats, width)
        self.input_proj1 = nn.Linear(self.embedder.out_dim + point_feats, width)

        self.cross_attn = ResidualCrossAttentionBlock(
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            data_width=None,
        )
        self.cross_attn1 = ResidualCrossAttentionBlock(
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            data_width=None,
        )

        self.self_attn = SelfAttentionTransformer(
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
        )

        if use_ln_post:
            self.ln_post = nn.LayerNorm(width)
        else:
            self.ln_post = nn.Identity()

    def forward(
        self,
        coarse_pc,
        sharp_pc,
        coarse_feats,
        sharp_feats,
        is_training,
        force_token_distribution=None,
    ):
        bs, N_coarse, D_coarse = coarse_pc.shape
        bs, N_sharp, D_sharp = sharp_pc.shape

        coarse_data = self.embedder(coarse_pc)

        if coarse_feats is not None:
            coarse_data = torch.cat([coarse_data, coarse_feats], dim=-1)

        coarse_data = self.input_proj(coarse_data)

        sharp_data = self.embedder(sharp_pc)

        if sharp_feats is not None:
            sharp_data = torch.cat([sharp_data, sharp_feats], dim=-1)

        sharp_data = self.input_proj1(sharp_data)

        if self.use_downsample:
            coarse_ratios = self.tokens / N_coarse
            sharp_ratios = self.tokens / N_sharp

            if force_token_distribution is not None:
                probabilities = force_token_distribution
            else:
                if not is_training:
                    probabilities = self.infer_probabilities
                elif is_training:
                    probabilities = self.train_probabilities

            ratio_coarse = np.random.choice(coarse_ratios, size=1, p=probabilities)[0]
            index = np.where(coarse_ratios == ratio_coarse)[0]
            ratio_sharp = sharp_ratios[index].item()

            flattened = coarse_pc.view(bs * N_coarse, D_coarse)
            batch = torch.arange(bs).to(coarse_pc.device)
            batch = torch.repeat_interleave(batch, N_coarse)
            pos = flattened
            idx = fps(pos, batch, ratio=ratio_coarse)
            query_coarse = coarse_data.view(bs * N_coarse, -1)[idx].view(
                bs, -1, coarse_data.shape[-1]
            )

            flattened = sharp_pc.view(bs * N_sharp, D_sharp)
            batch = torch.arange(bs).to(sharp_pc.device)
            batch = torch.repeat_interleave(batch, N_sharp)
            pos = flattened
            idx = fps(pos, batch, ratio=ratio_sharp)
            query_sharp = sharp_data.view(bs * N_sharp, -1)[idx].view(
                bs, -1, sharp_data.shape[-1]
            )

            query = torch.cat([query_coarse, query_sharp], dim=1)
        else:
            query = self.query
            query = repeat(query, "m c -> b m c", b=bs)

        latents_coarse = self.cross_attn(query, coarse_data)
        latents_sharp = self.cross_attn1(query, sharp_data)
        latents = latents_coarse + latents_sharp

        latents = self.self_attn(latents)
        latents = self.ln_post(latents)

        return latents


class PerceiverCrossAttentionDecoder(nn.Module):
    """Perceiver decoder: queries latents at 3D positions to get occupancy/UDF values."""

    def __init__(
        self,
        out_dim: int,
        embedder: FourierEmbedder,
        width: int,
        heads: int,
        init_scale: float,
        qkv_bias: bool,
    ):
        super().__init__()

        self.embedder = embedder
        self.query_proj = nn.Linear(self.embedder.out_dim, width)

        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            data_width=None,
        )
        self.ln_post = nn.LayerNorm(width)
        self.output_proj = nn.Linear(width, out_dim)

    def forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        queries = self.query_proj(self.embedder(queries))
        x = self.cross_attn_decoder(queries, latents)
        x = self.ln_post(x)
        x = self.output_proj(x)
        return x


class MichelangeloLikeAutoencoderWrapper:
    """Convenience wrapper for loading and using the pretrained 3D VAE."""

    def __init__(self, vae_ckpt_path, device):
        base_dir = os.path.dirname(vae_ckpt_path)
        yaml_file = os.path.join(base_dir, "vae-config.yaml")
        cfg = omegaconf.OmegaConf.load(yaml_file)
        self.model = (
            MichelangeloLikeAutoencoder(**cfg.vae).to(torch.bfloat16).to(device)
        )
        state_dict = torch.load(vae_ckpt_path, map_location=device, weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.use_udf_extraction = True
        self.model.udf_iso = 0.35
        self.model.eval()
        self.model = torch.compile(self.model, fullgraph=True)

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def encode_for_diffusion(
        self,
        coarse_surface,
        sharp_surface,
        is_training,
        force_token_distribution,
        sample_posterior,
    ):
        return self.model.encode_for_diffusion(
            coarse_surface,
            sharp_surface,
            is_training,
            force_token_distribution,
            sample_posterior,
        )

    def infer_mesh_from_latents(self, kl_embed, octree_depth=8):
        return self.model.infer_mesh_from_latents(kl_embed, octree_depth=octree_depth)

    def get_latent_code_count(self, index):
        return int(self.model.get_token_scales()[index])

    def get_embed_dim(self):
        return self.model.embed_dim


class MichelangeloLikeAutoencoder(AutoEncoder):
    """
    3D VAE with perceiver-style encoder/decoder.

    Encodes surface point clouds to latent tokens, decodes by querying at 3D positions.
    Supports variable token counts via FPS downsampling.
    """

    def __init__(
        self,
        use_downsample,
        num_latents,
        point_feats,
        out_dim,
        embed_dim,
        width,
        heads,
        num_encoder_layers,
        num_decoder_layers,
        token_scales,
        token_probability,
        init_scale=0.25,
        qkv_bias=False,
        use_ln_post=True,
        use_udf_extraction=False,
        schedule=None,  # not used
        force_scale=None,  # not used
    ):
        super().__init__(embed_dim=embed_dim, use_udf_extraction=use_udf_extraction)

        self.use_downsample = use_downsample
        self.num_latents = num_latents
        self.point_feats = point_feats
        self.out_dim = out_dim
        self.width = width
        self.heads = heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.init_scale = init_scale
        self.qkv_bias = qkv_bias
        self.use_ln_post = use_ln_post

        self.embedder = get_embedder(
            embed_type="fourier",
            num_freqs=8,
            include_pi=False,
        )

        self.init_scale = init_scale * math.sqrt(1.0 / self.width)

        self.encoder = SharpCoarseCrossAttentionEncoder(
            use_downsample=self.use_downsample,
            embedder=self.embedder,
            num_latents=self.num_latents,
            point_feats=self.point_feats,
            width=self.width,
            heads=self.heads,
            layers=self.num_encoder_layers,
            token_scales=token_scales,
            token_probability=token_probability,
            init_scale=self.init_scale,
            qkv_bias=self.qkv_bias,
            use_ln_post=self.use_ln_post,
        )

        if self.embed_dim > 0:
            # VAE embed
            self.pre_kl = nn.Linear(self.width, self.embed_dim * 2)
            self.post_kl = nn.Linear(self.embed_dim, self.width)
            self.latent_shape = (self.num_latents, self.embed_dim)
        else:
            self.latent_shape = (self.num_latents, self.width)

        self.transformer = SelfAttentionTransformer(
            width=self.width,
            layers=self.num_decoder_layers,
            heads=self.heads,
            init_scale=self.init_scale,
            qkv_bias=self.qkv_bias,
        )

        self.decoder = PerceiverCrossAttentionDecoder(
            embedder=self.embedder,
            out_dim=self.out_dim,
            width=self.width,
            heads=self.heads,
            init_scale=self.init_scale,
            qkv_bias=self.qkv_bias,
        )

    def get_token_scales(self):
        return self.encoder.tokens

    def set_inference_scale_probabilities(self, probabilities):
        assert len(probabilities) == len(self.encoder.tokens)
        self.encoder.infer_probabilities = probabilities

    def encode(self, coarse_surface, sharp_surface, sample_posterior, is_training):
        coarse_pc, coarse_feats = coarse_surface[..., :3], coarse_surface[..., 3:]
        sharp_pc, sharp_feats = sharp_surface[..., :3], sharp_surface[..., 3:]
        shape_latents = self.encoder(
            coarse_pc, sharp_pc, coarse_feats, sharp_feats, is_training
        )
        kl_embed, posterior = self.encode_kl_embed(shape_latents, sample_posterior)
        return shape_latents, kl_embed, posterior

    @torch.no_grad()
    def encode_for_diffusion(
        self,
        coarse_surface,
        sharp_surface,
        is_training,
        force_token_distribution,
        sample_posterior,
    ):
        coarse_pc, coarse_feats = coarse_surface[..., :3], coarse_surface[..., 3:]
        sharp_pc, sharp_feats = sharp_surface[..., :3], sharp_surface[..., 3:]
        shape_latents = self.encoder(
            coarse_pc,
            sharp_pc,
            coarse_feats,
            sharp_feats,
            is_training=is_training,
            force_token_distribution=force_token_distribution,
        )
        kl_embed, _ = self.encode_kl_embed(shape_latents, sample_posterior)
        return kl_embed

    def decode(self, z):
        latents = self.post_kl(z)
        return self.transformer(latents)

    def query(self, queries, latents):
        logits = self.decoder(queries, latents).squeeze(-1)
        return logits

    def encode_kl_embed(self, latents, sample_posterior):
        posterior = None
        if self.embed_dim > 0:
            moments = self.pre_kl(latents)
            posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)
            if sample_posterior:
                kl_embed = posterior.sample()
            else:
                kl_embed = posterior.mode()
        else:
            kl_embed = latents
        return kl_embed, posterior

    def forward(
        self, coarse_surface, sharp_surface, queries, sample_posterior, is_training
    ):
        shape_latents, kl_embed, posterior = self.encode(
            coarse_surface, sharp_surface, sample_posterior, is_training
        )
        latents = self.decode(kl_embed)
        logits = self.query(queries, latents)

        mean_value = torch.mean(kl_embed).detach()
        variance_value = torch.var(kl_embed).detach()

        return shape_latents, latents, posterior, logits, mean_value, variance_value

    @torch.no_grad()
    def infer_mesh(self, batch, octree_depth=8):
        _, kl_embed, _ = self.encode(
            batch["coarse_feats"], batch["sharp_feats"], True, False
        )
        latents = self.decode(kl_embed)
        return self.extract_mesh(latents, octree_depth=octree_depth)

    @torch.no_grad()
    def infer_mesh_from_latents(self, kl_embed, octree_depth=8):
        latents = self.decode(kl_embed)
        return self.extract_mesh(latents, octree_depth=octree_depth)
