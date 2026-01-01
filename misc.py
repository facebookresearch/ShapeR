# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

URL_DENOISER = "manifold://yawarnihal/tree/share/ShapeR/ase-itp-text_280-DiG-sflux-2v_1B/019-0-bfloat16.ckpt"
URL_DENOISER_CONFIG = "manifold://yawarnihal/tree/share/ShapeR/ase-itp-text_280-DiG-sflux-2v_1B/config.yaml"
URL_VAE = "manifold://yawarnihal/tree/share/ShapeR/ase-itp-text_280-DiG-sflux-2v_1B/vae-088-0-bfloat16.ckpt"
URL_VAE_CONFIG = "manifold://yawarnihal/tree/share/ShapeR/ase-itp-text_280-DiG-sflux-2v_1B/vae-config.yaml"
URL_DINO = "manifold://efm_public/tree/pretrained_weights/dinov2/dinov2_vitl14_reg4_pretrain.pth"

DL_OVERRIDE = {}


def setup_download_override(
    path_to_denoiser,
    path_to_denoiser_config,
    path_to_vae,
    path_to_vae_config,
    path_to_dino,
):
    global DL_OVERRIDE
    DL_OVERRIDE = {
        URL_DENOISER: path_to_denoiser,
        URL_DENOISER_CONFIG: path_to_denoiser_config,
        URL_VAE: path_to_vae,
        URL_VAE_CONFIG: path_to_vae_config,
        URL_DINO: path_to_dino,
    }


def get_local_path(manifold_path, recursive=False):
    if manifold_path in DL_OVERRIDE:
        print(f"Using override for {manifold_path} => {DL_OVERRIDE[manifold_path]}")
        assert os.path.exists(
            DL_OVERRIDE[manifold_path]
        ), f"{DL_OVERRIDE[manifold_path]} does not exist"
        return DL_OVERRIDE[manifold_path]
    print(f"Downloading from manifold {manifold_path}")
    sha = hashlib.sha256()
    if manifold_path.startswith("manifold://"):
        manifold_path = manifold_path.replace("manifold://", "")
    Path("/tmp/manifold_downloads").mkdir(parents=True, exist_ok=True)
    sha.update(manifold_path.encode())
    name = sha.hexdigest()[:16]
    basename = os.path.basename(manifold_path)
    if os.path.exists(f"/tmp/manifold_downloads/{name}"):
        return f"/tmp/manifold_downloads/{name}"
    with tempfile.TemporaryDirectory() as tmpdirname:
        dst = os.path.join(tmpdirname, name)
        print(f"Downloading from manifold {manifold_path} => {dst}")
        if recursive:
            if os.path.exists(dst):
                shutil.rmtree(dst)
            if os.path.exists(f"/tmp/manifold_downloads/{name}"):
                shutil.rmtree(f"/tmp/manifold_downloads/{name}")
            os.system(f"manifold getr --jobs 8 --threads 8 {manifold_path} {dst}")
            shutil.copytree(
                os.path.join(dst, basename), f"/tmp/manifold_downloads/{name}"
            )
            shutil.rmtree(dst)
        else:
            os.system(f"manifold get --threads 12 {manifold_path} {dst}")
            shutil.copyfile(dst, f"/tmp/manifold_downloads/{name}")
            os.remove(dst)
    return f"/tmp/manifold_downloads/{name}"
