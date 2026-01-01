# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import pickle

import time

import cv2 as cv
import numpy as np
import omegaconf
import torch
# important! We are using an old version of torchsparse, please use the fbcode version otherwise you will get errors,\
# since torchsparse changed their datastructures in newer versions

import trimesh

from misc import get_local_path
from model.flow_matching.shaper_denoiser import ShapeRDenoiser
from model.text.hf_embedder import TextFeatureExtractor
from model.vae3d.autoencoder import MichelangeloLikeAutoencoderWrapper

# @lint-ignore-every PYTHONPICKLEISBAD

from preprocessing.helper import (
    crop_and_resize,
    get_caption,
    get_parameters_from_state_dict,
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
        default="yawarnihal/tree/experiments/Native3D_FM/ase-itp-text_280-DiG-sflux-2v_1B/checkpoints/019-0.ckpt",
        help="Path to the checkpoint file.",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="yawarnihal/tree/experiments/Native3D_FM/ase-itp-text_280-DiG-sflux-2v_1B/config.yaml",
        help="Path to the config yaml.",
    )
    parser.add_argument(
        "--input_pkl",
        type=str,
        default="yawarnihal/tree/tmp/livestack_dumps/61190c910df4+_2025-11-26-11-30-39/fd11fd99-31c8-d930-5f2f-e4f91fd0f739.pkl",
        help="Path to the input pkl file which contains semidense points and the bounds.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="mesh.obj",
        help="Path to the output obj file.",
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
    args = parser.parse_args()

    # load the checkpoint
    if not args.ckpt_file.startswith("manifold://"):
        args.ckpt_file = "manifold://" + args.ckpt_file

    t = time.time()
    print("Loading checkpoint from", args.ckpt_file)
    ckpt_file = get_local_path(args.ckpt_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]

    print("Loading config from", args.config_file)
    # load the config (usually located in the folder above checkpoint)
    if not args.config_file.startswith("manifold://"):
        args.config_file = "manifold://" + args.config_file
    yaml_file = get_local_path(args.config_file)
    config = omegaconf.OmegaConf.load(yaml_file)

    print(f"-- Loading Model files in {time.time() - t} seconds")
    t = time.time()

    # load the model and weights
    print("Loading model...")
    model = ShapeRDenoiser(config).to(device)
    model.load_state_dict(
        get_parameters_from_state_dict(state_dict, "model"), strict=False
    )
    vae = MichelangeloLikeAutoencoderWrapper(config.resume_vae).to(device)
    text_feature_extractor = TextFeatureExtractor(device=device)

    print(f"-- Init model in {time.time() - t} seconds")
    t = time.time()

    print("Converting to bfloat16...")
    vae = vae.to(torch.bfloat16)
    text_feature_extractor = text_feature_extractor.to(torch.bfloat16)
    model.convert_to_bfloat16()
    model = torch.compile(model, fullgraph=True)
    model = model.eval()

    print(f"-- Model to b16 & compile in {time.time() - t} seconds")
    t = time.time()

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
    pk = pickle.load(open(get_local_path(args.input_pkl), "rb"))

    print(f"-- Loading input in {time.time() - t} seconds")
    t = time.time()

    mask_ingests = []
    for image_idx in range(pk["images"][0].shape[0]):
        im_mask = project_point_to_image(
            pk["semi_dense_points_orig"].cpu().numpy(),
            pk["camera_intrinsics"][image_idx].cpu().numpy(),
            pk["camera_extrinsics"][image_idx].cpu().numpy(),
            pk["images"][0][image_idx].shape[-1],
            pk["images"][0][image_idx].shape[-2],
        )
        im_mask = cv.dilate(im_mask, np.ones((3, 3), np.uint8), iterations=2)
        mask_ingests.append(im_mask[None, None, None, :, :])
    mask_ingests = np.concatenate(mask_ingests, axis=1)
    mask_ingests = torch.from_numpy(mask_ingests).to(device).float() / 255.0
    batch = {
        "name": [pk["name"]],
        "semi_dense_points": pk["semi_dense_points"].to(device),
        "images": pk["images"].to(device).to(torch.bfloat16),
        "masks_ingest": mask_ingests.to(torch.bfloat16),
        "camera_extrinsics": pk["camera_extrinsics"]
        .unsqueeze(0)
        .to(device)
        .to(torch.bfloat16),
        "camera_intrinsics": pk["camera_intrinsics"]
        .unsqueeze(0)
        .to(device)
        .to(torch.bfloat16),
        "caption": [get_caption(pk)],
        "boxes_ingest": torch.zeros(1, pk["images"].shape[1], 2, 2)
        .to(device)
        .to(torch.bfloat16),
    }

    print(f"-- Batch prepare in {time.time() - t} seconds")
    t = time.time()
    # check mask ingests

    # at this point we have the points, images, masks, camera intrinsics and extrinsics
    # and can proceed to inference

    # Part 2: Inference of mesh

    with torch.no_grad():
        latents_pred = model.infer_latents(
            batch,
            token_shape=token_shape,
            text_feature_extractor=text_feature_extractor,
            num_steps=25,
            use_shifted_sampling=use_shifted_sampling,
        )
        mesh = vae.infer_mesh_from_latents(latents_pred)[0]
        print(f"-- Inference in {time.time() - t} seconds")
        t = time.time()
        if args.remove_floating_geometry:
            mesh = remove_floating_geometry(mesh)
        # simplify the mesh otherwise it will be too large if you mesh it at 128x128x128 resolution
        if args.simplify_mesh:
            mesh = mesh.simplify_quadric_decimation(face_count=75000)
        mesh.export(args.output_path)
        print(f"-- Mesh export in {time.time() - t} seconds")


if __name__ == "__main__":
    main()
