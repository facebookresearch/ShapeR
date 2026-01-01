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
        "--clean-cache",
        action="store_true",
        help="Clean manifold cache.",
    )
    parser.add_argument(
        "--test_local_weight_override",
        action="store_true",
        help="Test with local weights override",
    )
    args = parser.parse_args()

    if args.clean_cache:
        if os.path.exists("/tmp/manifold_downloads"):
            shutil.rmtree("/tmp/manifold_downloads", ignore_errors=True)

    if args.test_local_weight_override:
        # example override of weights stored in /home/yawarnihal/shaper_weights
        setup_download_override(
            "/home/yawarnihal/shaper_weights/019-0-bfloat16.ckpt",
            "/home/yawarnihal/shaper_weights/config.yaml",
            "/home/yawarnihal/shaper_weights/vae-088-0-bfloat16.ckpt",
            "/home/yawarnihal/shaper_weights/vae-config.yaml",
            "/home/yawarnihal/shaper_weights/dinov2_vitl14_reg4_pretrain.pth",
            "/home/yawarnihal/shaper_weights/t5-v1_1-xl",
            "/home/yawarnihal/shaper_weights/clip-vit-large-patch14",
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

    print("preprocessing inputs")
    # Part 1a: Data processing of point clouds

    semi_dense_points = pkl_sample["points"].clone()
    bounds = pkl_sample["halfBounds"]
    scale = 0.9 / bounds.max()
    semi_dense_points[:, :3] *= scale

    assert semi_dense_points.shape[0] >= 16, "give atleast 16 points"

    # Part 1b: Data processing of images and camera parameters

    # view selection strategy: take the last N views
    # IMPORTANT TODO: Dont choose images which have object close to the edge, they get messed up on rectification!

    crops, masks, camera_params, Ts_camera_model, paddedCropsXYWHC = (
        view_angle_based_strategy(
            pkl_sample["crops"],
            pkl_sample["masks"],
            pkl_sample["camera_params"],
            pkl_sample["Ts_camera_model"],
            pkl_sample["paddedCropsXYWHC"],
            args.num_input_images,
            pkl_sample.get("isNebula", False),
        )
    )

    print("Using", len(crops), "images for inference")

    # idx=4; fullImages = pkl_sample['fullImages'][-16:]; arr = project_point_to_image_with_distortion(pkl_sample["points"].cpu().numpy(), Ts_camera_model[idx].cpu().numpy(), camera_params[idx].cpu().numpy(), [512, 512]); img = np.zeros([512, 512, 3]).astype(np.uint8); img[:, :, 0] = fullImages[idx, :, :, 0].cpu().numpy(); img[:, :, 1] = arr; Image.fromarray(img).save(f'dist_projected_image_{idx}.jpg')

    # calculate extrinsics
    camera_extrinsics = torch.linalg.inv(Ts_camera_model)
    camera_extrinsics[:, :3, 3] = camera_extrinsics[:, :3, 3] * scale

    # rectify images
    crops, masks = pad_for_rectification(
        crops, masks, paddedCropsXYWHC, pkl_sample.get("isNebula", False)
    )
    crops, masks, camera_params = rectify_images(crops, masks, camera_params)

    # idx = 4;mask = project_point_to_image(semi_dense_points.cpu().numpy(), camera_params[idx].cpu().numpy(), camera_extrinsics[idx].cpu().numpy(), crops[idx].shape[1], crops[idx].shape[0]); img = (crops[idx].unsqueeze(0).repeat((3, 1, 1)).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8); img[:, :, 1] = mask; Image.fromarray(img).save(f"projected_points_{idx}.jpg")

    # get new crops with rectified images
    crops, camera_params = crop_and_resize(
        crops, masks, camera_params, target_size=config.encoder.dino_image_size
    )

    if pkl_sample.get("isNebula", False):
        # rotate the image ccw
        for im_idx in range(crops.shape[0]):
            crops[im_idx] = torch.rot90(crops[im_idx], 1, [1, 2])

            camera_params[im_idx] = rotate_intrinsics_ccw90(
                camera_params[im_idx], crops[im_idx].shape[2]
            )
            camera_extrinsics[im_idx] = rotate_extrinsics_ccw90(
                camera_extrinsics[im_idx]
            )

    # check projections
    # idx = 4;mask = project_point_to_image(semi_dense_points.cpu().numpy(), camera_params[idx].cpu().numpy(), camera_extrinsics[idx].cpu().numpy(), crops[idx].shape[2], crops[idx].shape[1]); img = (crops[idx].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8); img[:, :, 1] = mask; Image.fromarray(img).save(f"projected_points_{idx}.jpg")

    mask_ingests = []
    for image_idx in range(crops.shape[0]):
        im_mask = project_point_to_image(
            semi_dense_points.cpu().numpy(),
            camera_params[image_idx].cpu().numpy(),
            camera_extrinsics[image_idx].cpu().numpy(),
            crops[image_idx].shape[-1],
            crops[image_idx].shape[-2],
        )
        im_mask = cv.dilate(im_mask, np.ones((3, 3), np.uint8), iterations=2)
        mask_ingests.append(im_mask[None, None, None, :, :])

    mask_ingests = np.concatenate(mask_ingests, axis=1)
    mask_ingests = torch.from_numpy(mask_ingests).to(device).float() / 255.0
    boxes_ingest = torch.zeros(1, crops.shape[0], 2, 2).to(device)

    # check mask ingests

    # at this point we have the points, images, masks, camera intrinsics and extrinsics
    # and can proceed to inference

    # Part 2: Inference of mesh

    print("Inferring multiple times for benchmarking...")
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
