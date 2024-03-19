# StratXplore-VLN

## Get started


## License

The chosen license in accordance with legal department must be defined into an explicit [LICENSE](https://github.com/ThalesGroup/template-project/blob/master/LICENSE) file at the root of the repository
You can also link this file in this README section.
## TODOs

* [X] Release VLN (R2R, R4R) code.
* [X] Data preprocessing code.
* [] Release checkpoints and preprocessed datasets.

## Setup

### Installation

1. Create a virtual environment. We develop this project with Python 3.6.

   ```bash
   conda env create -f environment.yaml
   ```
1. Install the latest version of [Matterport3DSimulator](https://github.com/peteanderson80/Matterport3DSimulator), 
including the Matterport3D RGBD datasets (for step 6).
1. Download the Matterport3D scene meshes. `download_mp.py` must be obtained from the Matterport3D [project webpage]
   (https://niessner.github.io/Matterport/). `download_mp.py` is also used for downloading RGBD datasets in step 2.

```bash
# run with python 2.7
python download_mp.py --task habitat -o data/scene_datasets/mp3d/
# Extract to: ./data/scene_datasets/mp3d/{scene}/{scene}.glb
```

1. Grid feature preprocessing for metric mapping (~100G).

   ```bash
   # for R2R, RxR, REVERIE
   python precompute_features/grid_mp3d_clip.py
   python precompute_features/grid_mp3d_imagenet.py
   python precompute_features/grid_depth.py
   python precompute_features/grid_sem.py

   # for R2R-CE pre-training
   python precompute_features/grid_habitat_clip.py
   python precompute_features/save_habitat_img.py --img_type depth
   python precompute_features/save_depth_feature.py
   ```
1. Download preprocessed instruction datasets and trained weights [[link]](https://drive.google.
com/file/d/1jYg_dMlCDZoOtrkmmq40k-_-m6xerdUI/view?usp=sharing). The directory structure has been organized. For R2R-CE experiments, follow [ETPNav](https://github.com/MarSaKi/ETPNav) to configure VLN-CE datasets in `bevbert_ce/data` foler, and put the trained CE weights [[link]](https://drive.google.com/file/d/1-2u1NWmwpX09Rg7uT5mABo-CBTsLthGm/view?usp=sharing) in `bevbert_ce/ckpt`.

Good luck on your VLN journey with BEVBert!

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

# Acknowledge

Our implementations are partially inspired by VLN-BEVBERT, [DUET](https://github.com/cshizhe/VLN-DUET), [S-MapNet]
(https://github.com/vincentcartillier/Semantic-MapNet) and [ETPNav](https://github.com/MarSaKi/ETPNav).

Thank them for open sourcing their great works!
