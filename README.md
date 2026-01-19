## ShapeR: Robust Conditional 3D Shape Generation from Casual Captures

![teaser](resources/Teaser.jpg)

ShapeR introduces a novel approach to metric shape generation. Given an input image sequence, preprocessing extracts per-object metric sparse SLAM points, images, poses, and captions using off-the-shelf methods. A rectified flow transformer operating on VecSet latents conditions on these multimodal inputs to generate a shape code, which is decoded into the object’s mesh. By applying the model object-centrically to each detected object, we obtain a metric reconstruction of the entire scene.

[Project Page](http://facebookresearch.github.io/ShapeR)  |  [Paper](https://cdn.jsdelivr.net/gh/facebookresearch/ShapeR@main/resources/ShapeR.pdf)  |  [Video](https://www.youtube.com/watch?v=EbY30KAA55I)  |  [HF-Model](https://huggingface.co/facebook/ShapeR/)  |  [HF Evaluation Dataset](https://huggingface.co/datasets/facebook/ShapeR-Evaluation)

## Installation

Refer to [INSTALL.md](INSTALL.md) for detailed instructions on setting up the environment.

## Usage

### Inference

```bash
python infer_shape.py --input_pkl <sample.pkl> --config balance --output_dir output
```

**Arguments:**
- `--input_pkl`: Path to preprocessed pickle file (relative to `data/`)
- `--config`: Preset configuration
  - `quality`: 16 views, 50 steps (best quality, slowest)
  - `balance`: 16 views, 25 steps (recommended)
  - `speed`: 4 views, 10 steps (fastest)
- `--output_dir`: Output directory for meshes and visualizations
- `--do_transform_to_world`: Transform output mesh to world coordinates
- `--remove_floating_geometry`: Remove disconnected mesh components (default: on)
- `--simplify_mesh`: Reduce mesh complexity (default: on)
- `--save_visualization`: Save comparison visualization (default: off)

### Example

```bash
python infer_shape.py --input_pkl ADT1292__stool.pkl --config balance
```

**Output:**
- `<name>.glb`: Reconstructed 3D mesh
- `VIS__<name>.jpg`: Visualization comparing input, prediction, and ground truth (--save_visualization is passed)

![Example visualization](resources/output_vis_example.jpg)

## Data Format

This codebase assumes that a sequence has already been processed using the Aria MPS pipeline, along with an 3D object instance detector, resulting in the pickle files with the preprocessed data required for ShapeR ingestion.

We release the **ShapeR Evaluation Dataset** containing preprocessed samples from Aria glasses captures. Each sample is a pickle file with point clouds, multi-view images, camera parameters, text captions, and ground truth meshes.

For a detailed walkthrough of the data format, see the **[`explore_data.ipynb`](explore_data.ipynb)** notebook which includes:
- Complete pickle file structure with all keys and their dimensions
- Interactive 3D visualization of point clouds and meshes
- Camera position visualization
- Image and mask grid displays
- DataLoader usage examples for both SLAM and RGB variants
- Explanation of view selection strategies

## Project Structure

```
ShapeR/
├── infer_shape.py          # Main inference script
├── explore_data.ipynb      # Data exploration notebook
├── dataset/
│   ├── shaper_dataset.py   # Dataset and dataloader
│   ├── image_processor.py  # View selection and image preprocessing
│   └── point_cloud.py      # Point cloud to SparseTensor
├── model/
│   ├── flow_matching/
│   │   └── shaper_denoiser.py  # Flow matching denoiser
│   ├── vae3d/
│   │   └── autoencoder.py      # 3D VAE for mesh generation
│   ├── pointcloud_encoder.py   # Sparse 3D convolution encoder
│   └── dino_and_ray_feature_extractor.py  # Image feature extraction
├── preprocessing/
│   └── helper.py           # Fisheye rectification, camera utils
├── postprocessing/
│   └── helper.py           # Mesh cleanup, visualization
├── checkpoints/            # Model weights (downloaded automatically)
└── data/                   # Input pickle files
```

## License

The majority of ShapeR is licensed under CC-BY-NC. See the [LICENSE](LICENSE) file for details. However portions of the project are available under separate license terms: see [NOTICE](NOTICE).

## Citation

If you find ShapeR useful for your research, please cite our paper:

```bibtex
@misc{siddiqui2026shaperrobustconditional3d,
      title={ShapeR: Robust Conditional 3D Shape Generation from Casual Captures}, 
      author={Yawar Siddiqui and Duncan Frost and Samir Aroudj and Armen Avetisyan and Henry Howard-Jenkins and Daniel DeTone and Pierre Moulon and Qirui Wu and Zhengqin Li and Julian Straub and Richard Newcombe and Jakob Engel},
      year={2026},
      eprint={2601.11514},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.11514}, 
}
```
