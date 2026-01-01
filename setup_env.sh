#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# this needs python 3.10
# instructions are as follows:
# conda create -n lrm python=3.10
# conda activate lrm
# source setup_env.sh

# this will setup the enviroment in your OD for running the LRM
# making it possible to run the LRM outside of the fbsource repo
# create the conda environment and activate it first!


# important! this is the path to the cuda install for me on my OD.
# Might be different for you depending on your setup

USER=$(whoami)

export CUDA_INCLUDE=/home/${USER}/fbsource/third-party/cuda/cuda_12.8.0/x64-linux/include_no_implicit
export CUDA_HOME=/home/${USER}/fbsource/third-party/cuda/cuda_12.8.0/x64-linux
export CUDA_LIB=/home/${USER}/fbsource/third-party/cuda/cuda_12.8.0/x64-linux/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_LIB
export LIBRARY_PATH=$LIBRARY_PATH:$CUDA_LIB
export CFLAGS="-I$CUDA_HOME/include_no_implicit"
export CXXFLAGS="-I$CUDA_HOME/include_no_implicit"
export CPATH="$CUDA_HOME/include_no_implicit:$CPATH"
# path to where sparsehash is installed
export SPHASH_PATH=/home/${USER}/sphash
export SPHASH_INCLUDE=$SPHASH_PATH/include

install_sparsehash() {
  # assume proxy is set up
  CURRENT_DIR=$(pwd)
  cd ~
  git clone https://github.com/sparsehash/sparsehash.git
  cd sparsehash
  ./configure
  make
  make install exec_prefix=$SPHASH_PATH prefix=$SPHASH_PATH
  cd $CURRENT_DIR
}

install_packages() {
  export HTTPS_PROXY=http://fwdproxy:8080
  export HTTP_PROXY=http://fwdproxy:8080
  export FTP_PROXY=http://fwdproxy:8080
  export https_proxy=http://fwdproxy:8080
  export http_proxy=http://fwdproxy:8080
  export ftp_proxy=http://fwdproxy:8080
  export http_no_proxy='\''*.facebook.com|*.tfbnw.net|*.fb.com'\'
  python -m pip install wheel setuptools
  python -m pip install numpy tqdm hydra-core matplotlib opencv-python imageio easydict munch plyfile
  python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
  python -m pip install transformers trimesh scikit-image diffusers gradio peft einops
  python -m pip install flash-attn --no-build-isolation --no-cache-dir
  python -m pip install imageio\[ffmpeg\] imageio\[pyav\]
  python -m pip install pymeshlab sophuspy fast_simplification scikit-learn timm plotly torchdiffeq sentencepiece protobuf
  # if you have root access the following two steps become easier
  # (just install libsparseshash-dev and then torchsparse 1.4 from the official repo OR whichever way you install torchsparse for the old LRM)
  # but since we assume OD, we have to do it this way; this takes a while, be patient
  install_sparsehash
  pip install torch-cluster -f https://data.pyg.org/whl/torch-2.7.1+cu128.html
  python -m pip install --verbose git+https://github.com/nihalsid/torchsparse@legacy --no-build-isolation
}

install_packages
