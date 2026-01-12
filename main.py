# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import time

from pathlib import Path

import numpy as np
import omegaconf
import torch

# important! We are using an old version of torchsparse, please use the legacy version otherwise you will get errors,\
# since torchsparse changed their datastructures in newer versions

import trimesh

from dataset.download import setup_data
from dataset.shaper_dataset import InferenceDataset
from model.download import setup_checkpoints
from model.flow_matching.shaper_denoiser import ShapeRDenoiser
from model.text.hf_embedder import TextFeatureExtractor
from model.vae3d.autoencoder import MichelangeloLikeAutoencoderWrapper
from preprocessing.helper import remove_floating_geometry
from tqdm import tqdm

# @lint-ignore-every PYTHONPICKLEISBAD

preset_configs = {
    "quality": (16, 4, 50),
    "speed": (4, 2, 10),
    "balance": (16, 4, 25),
}


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_pkl",
        type=str,
        default="ADT_seq140M1292__BlackBarStool_B.pkl",
        help="Path to the input pkl file which contains the processed observations for the detected bbox.",
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
        "--output_dir",
        type=str,
        default="output",
        help="Path to the output mesh.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="balance",
        help="Config to use for the inference.",
    )
    parser.add_argument(
        "--do_transform_to_world",
        action="store_true",
        help="Transform the mesh to world coordinates.",
    )

    args = parser.parse_args()

    num_images, token_multiplier, num_steps = preset_configs[args.config]

    # example override of weights stored in /home/yawarnihal/shaper_weights
    # todo: once the checkpoints are on huggingface, adjust this
    setup_checkpoints()
    setup_data()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # load the checkpoint
    ckpt_file = "checkpoints/019-0-bfloat16.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(ckpt_file, map_location=device, weights_only=False)

    # load the config (usually located in the folder above checkpoint)
    yaml_file = "checkpoints/config.yaml"
    config = omegaconf.OmegaConf.load(yaml_file)
    # load the model and weights
    print("Loading model...")
    model = ShapeRDenoiser(config).to(device)
    model.convert_to_bfloat16()
    model.load_state_dict(state_dict, strict=False)

    vae = MichelangeloLikeAutoencoderWrapper(
        "checkpoints/vae-088-0-bfloat16.ckpt", device
    )

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
    token_count = int(scales[np.argmax(scale_prob)].item()) * token_multiplier
    token_shape = (1, token_count, vae.get_embed_dim())
    use_shifted_sampling = (
        getattr(config.fm_transformer, "time_sampler", "lognorm") == "flux"
    )

    # create batch sample
    print("Loading input pkl from", args.input_pkl)
    inference_dataset = InferenceDataset(
        config,
        paths=[os.path.join("data", args.input_pkl)],
        override_num_views=num_images,
    )
    inference_loader = torch.utils.data.DataLoader(
        inference_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=inference_dataset.custom_collate,
    )
    with torch.no_grad():
        for batch in tqdm(inference_loader):
            batch = InferenceDataset.move_batch_to_device(
                batch, device, dtype=torch.bfloat16
            )
            latents_pred = model.infer_latents(
                batch,
                token_shape=token_shape,
                text_feature_extractor=text_feature_extractor,
                num_steps=num_steps,
                use_shifted_sampling=use_shifted_sampling,
            )
            mesh = vae.infer_mesh_from_latents(latents_pred)[0]
            # remove floating geometry, keeping only the largest component
            # sometimes not the best way, but usually works out okay most of the time
            if args.remove_floating_geometry:
                mesh = remove_floating_geometry(mesh)
            # simplify the mesh otherwise it will be too large if you mesh it at 128x128x128 resolution
            if args.simplify_mesh:
                mesh = mesh.simplify_quadric_decimation(face_count=125000)
            # rescale back to the original scale
            mesh = inference_dataset.rescale_back(
                batch["index"][0], mesh, args.do_transform_to_world
            )
            tmp_output_path_mesh = "/tmp/mesh.obj"
            mesh.export(tmp_output_path_mesh)
            # convert to glb
            mesh = trimesh.load(tmp_output_path_mesh, force="mesh")
            mesh.export(output_dir / (batch["name"][0] + ".glb"), include_normals=True)


if __name__ == "__main__":
    main()
