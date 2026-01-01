# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import pickle
import shutil

import time

import cv2 as cv
import numpy as np
import omegaconf
import torch

# important! We are using an old version of torchsparse, please use the fbcode version otherwise you will get errors,\
# since torchsparse changed their datastructures in newer versions

import trimesh

from misc import get_local_path, setup_download_override
from model.flow_matching.shaper_denoiser import ShapeRDenoiser
from model.text.hf_embedder import TextFeatureExtractor
from model.vae3d.autoencoder import MichelangeloLikeAutoencoderWrapper

# @lint-ignore-every PYTHONPICKLEISBAD

from preprocessing.helper import (
    crop_and_resize,
    get_caption,
    pad_for_rectification,
    preprocess_point_cloud,
    project_point_to_image,
    rectify_images,
    remove_floating_geometry,
    rotate_extrinsics_ccw90,
    rotate_intrinsics_ccw90,
)
from preprocessing.view_selection_heuristic import view_angle_based_strategy


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_file",
        type=str,
        default="yawarnihal/tree/share/ShapeR/ase-itp-text_280-DiG-sflux-2v_1B/019-0-bfloat16.ckpt",
        help="Path to the checkpoint file.",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="yawarnihal/tree/share/ShapeR/ase-itp-text_280-DiG-sflux-2v_1B/config.yaml",
        help="Path to the config yaml.",
    )
    parser.add_argument(
        "--input_pkl",
        type=str,
        default="danpb/tree/tmp_share/image_lrm_input/nebula/0000000001.pkl",
        help="Path to the input pkl file which contains semidense points and the bounds.",
    )
    parser.add_argument(
        "--num_input_images",
        type=int,
        default=16,
        help="Path to the input pkl file which contains semidense points and the bounds.",
    )
    parser.add_argument(
        "--remove_floating_geometry",
        action="store_false",
        help="Remove floating geometry from the mesh.",
    )
    parser.add_argument(
        "--simplify_mesh",
        action="store_false",
        help="Simplify the mesh.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="mesh.glb",
        help="Path to the output mesh.",
    )
    parser.add_argument(
        "--test_local_weight_override",
        action="store_true",
        help="Test with local weights override",
    )
    args = parser.parse_args()

    if args.test_local_weight_override:
        # example override of weights stored in /home/yawarnihal/shaper_weights
        setup_download_override(
            "/home/yawarnihal/shaper_weights/019-0-bfloat16.ckpt",
            "/home/yawarnihal/shaper_weights/config.yaml",
            "/home/yawarnihal/shaper_weights/vae-088-0-bfloat16.ckpt",
            "/home/yawarnihal/shaper_weights/vae-config.yaml",
            "/home/yawarnihal/shaper_weights/dinov2_vitl14_reg4_pretrain.pth",
        )

    # load the checkpoint
    if not args.ckpt_file.startswith("manifold://"):
        args.ckpt_file = "manifold://" + args.ckpt_file
    print("Loading checkpoint from", args.ckpt_file)
    ckpt_file = get_local_path(args.ckpt_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(ckpt_file, map_location=device, weights_only=False)

    print("Loading config from", args.config_file)
    # load the config (usually located in the folder above checkpoint)
    if not args.config_file.startswith("manifold://"):
        args.config_file = "manifold://" + args.config_file
    yaml_file = get_local_path(args.config_file)
    config = omegaconf.OmegaConf.load(yaml_file)

    # load the model and weights
    print("Loading model...")
    model = ShapeRDenoiser(config).to(device)
    model.convert_to_bfloat16()
    model.load_state_dict(state_dict, strict=False)

    vae = MichelangeloLikeAutoencoderWrapper(config.resume_vae, device)

    text_feature_extractor = TextFeatureExtractor(device=device)
    text_feature_extractor = text_feature_extractor.to(torch.bfloat16)

    model = torch.compile(model, fullgraph=True)
    model = model.eval()
    vae.model.use_udf_extraction = True
    vae.model.udf_iso = 0.375

    scales = vae.model.get_token_scales()
    scale_prob = np.zeros_like(scales)
    scale_prob[6] = 1.0
    vae.model.set_inference_scale_probabilities(scale_prob)
    token_count = int(scales[np.argmax(scale_prob)].item()) * 4
    token_shape = (1, token_count, vae.get_embed_dim())
    use_shifted_sampling = (
        getattr(config.fm_transformer, "time_sampler", "lognorm") == "flux"
    )

    # create batch sample
    if not args.input_pkl.startswith("manifold://"):
        args.input_pkl = "manifold://" + args.input_pkl

    print("Loading input pkl from", args.input_pkl)
    pkl_sample = pickle.load(open(get_local_path(args.input_pkl), "rb"))

    for ctr in range(5):
        print("-" * 20)
        print(f"Running inference {ctr+1}...")

        # create torchsparse tensor
        batch_sdp = preprocess_point_cloud(
            [semi_dense_points],
            num_bins=config.encoder.num_bins,
        )

        batch = {
            "index": [0],
            "name": ["test"],
            "semi_dense_points": batch_sdp,
            "images": crops.unsqueeze(0).to(torch.bfloat16),
            "masks_ingest": mask_ingests.to(torch.bfloat16),
            "camera_extrinsics": camera_extrinsics.unsqueeze(0).to(torch.bfloat16),
            "camera_intrinsics": camera_params.unsqueeze(0).to(torch.bfloat16),
            "boxes_ingest": boxes_ingest.to(torch.bfloat16),
            "caption": [get_caption(pkl_sample)],
        }

        with torch.no_grad():
            t_inference = time.time()
            latents_pred = model.infer_latents(
                batch,
                token_shape=token_shape,
                text_feature_extractor=text_feature_extractor,
                num_steps=25,
                use_shifted_sampling=use_shifted_sampling,
            )
            mesh = vae.infer_mesh_from_latents(latents_pred)[0]
            print(f"inference took {time.time() - t_inference} seconds")
            # remove floating geometry, keeping only the largest component
            # sometimes not the best way, but usually works out okay most of the time
            if args.remove_floating_geometry:
                mesh = remove_floating_geometry(mesh)
            # simplify the mesh otherwise it will be too large if you mesh it at 128x128x128 resolution
            if args.simplify_mesh:
                mesh = mesh.simplify_quadric_decimation(face_count=75000)
            # rescale back to the original scale
            bounds = pkl_sample["halfBounds"].cpu().numpy()
            scale = 0.9 / np.max(bounds)
            mesh.apply_scale(1 / scale)
            tmp_output_path_mesh = "/tmp/mesh.obj"
            mesh.export(tmp_output_path_mesh)
            # convert to glb
            mesh = trimesh.load(tmp_output_path_mesh, force="mesh")
            if not args.output_path.endswith(".glb"):
                args.output_path += ".glb"
            mesh.export(args.output_path, include_normals=True)


if __name__ == "__main__":
    main()
