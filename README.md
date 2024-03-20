# StratXplore-VLN
We introduce a memory-based and mistake-aware path planning strategy for VLN agents, called *StratXplore* (Strategic 
Exploration), 
that presents global and local action planning to select optimal frontier for path correction.
## Get started


## TODOs

* [X] Release VLN (R2R, R4R) code.
* [X] Data preprocessing code.
* [ ] Release visualisation code.
* [ ] Release checkpoints and preprocessed datasets.

## Setup

### Installation

1. Create a virtual environment. We develop this project with Python 3.6.

   ```bash
   conda env create -f environment.yaml
   ```
2. Install the latest version of [Matterport3DSimulator](https://github.com/peteanderson80/Matterport3DSimulator), 
including the Matterport3D RGBD datasets (for step 6).
3. Download the Matterport3D scene meshes. `download_mp.py` must be obtained from the Matterport3D [project webpage]
   (https://niessner.github.io/Matterport/). `download_mp.py` is also used for downloading RGBD datasets in step 2.

   ```bash
   # run with python 2.7
   python download_mp.py --task habitat -o data/scene_datasets/mp3d/
   # Extract to: ./data/scene_datasets/mp3d/{scene}/{scene}.glb
   ```

4. Grid feature preprocessing for metric mapping (~100G).

   ```bash
   # for R2R, R4R
   python precompute_features/grid_mp3d_clip.py
   python precompute_features/grid_mp3d_imagenet.py
   python precompute_features/grid_depth.py
   python precompute_features/grid_sem.py
   ```
5. Download preprocessed instruction datasets and trained weights [[link]](https://drive.google.
com/file/d/1jYg_dMlCDZoOtrkmmq40k-_-m6xerdUI/view?usp=sharing). The directory structure has been organized.

## Running

   Pre-training. Download precomputed image features [[link]](https://drive.google.com/file/d/1S8jD1Mln0mbTsB5I_i2jdQ8xBbnw-Dyr/view?usp=sharing) into folder `img_features`.

   ```
   CUDA_VISIBLE_DEVICES=0 bash scripts/pt_r2r.bash 2333  # R2R
   ```

   Fine-tuning and Testing, the trained weights can be found in step 7.

   ```
   CUDA_VISIBLE_DEVICES=0 bash scripts/ft_r2r.bash 2333  # R2R
   ```

# Contact Information

* k DOT gopinathan AT ecu DOT edu DOT au

# Acknowledgement

Our implementations are partially inspired by VLN-BEVBERT, [DUET](https://github.com/cshizhe/VLN-DUET), [S-MapNet]
(https://github.com/vincentcartillier/Semantic-MapNet) and [ETPNav](https://github.com/MarSaKi/ETPNav).

Thank them for open sourcing their great works!
