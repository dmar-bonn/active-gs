# ActiveGS: Active Scene Reconstruction Using Gaussian Splatting

Liren Jin<sup>1</sup>, Xingguang Zhong<sup>1</sup>, Yue Pan<sup>1</sup>, Jens Behley<sup>1</sup>, Cyrill Stachniss<sup>1</sup>, Marija Popovic<sup>2</sup> 

<sup>1</sup> University of Bonn, <sup>2</sup> TU Delft

```commandline
@article{jin2025ral,
author      = {Jin, Liren and Zhong, Xingguang and Pan, Yue and Behley, Jens and Stachniss, Cyrill and Popović, Marija},
journal     = {IEEE Robotics and Automation Letters}, 
title       = {{ActiveGS: Active Scene Reconstruction Using Gaussian Splatting}}, 
year        = {2025},
volume      = {10},
number      = {5},
pages       = {4866-4873},
doi         = {10.1109/LRA.2025.3555149}}
```

<a target="_blank" href="https://arxiv.org/abs/2412.17769">
    <img src="https://img.shields.io/badge/arXiv-2412.17769-b31b1b.svg" alt="arXiv Paper">
</a>


![teaser](https://github.com/user-attachments/assets/f7c16143-fc3c-4d69-88f3-f7bd43d9de85)
## Setup
We test the following setup on Ubuntu20 with CUDA11.8. 

Clone ActiveGS repo:
```
git clone git@github.com:dmar-bonn/active-gs.git
cd active-gs
```

(optional) For different CUDA versions in your machine, you might need to change the corresponding pytorch version and source in envs/build.sh:
```
# for example for CUDA 12.1, change the source. 
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# you can find more compatible version on https://pytorch.org/get-started/previous-versions/
```
Create and activate environment:
```
bash envs/build.sh
conda activate active-gs
```

## Data

Download full Replica dataset:
```
bash data/replica_download.sh
```

(optional) If you only want to quickly try one example, rather than the whole dataset, use:
```
bash data/replica_example.sh
```
This will only download office0 scene.

## Run
Run online mission:
```
python main.py planner=PLANNER_TYPE scene=SCENE_NAME
# example: 
# python main.py planner=confidence scene=replica/office0 use_gui=true
```
If use_gui is set to true, you should be able to see a GUI running.


To visualize the built GS map:
```
python visualize.py -G PATH_TO_GS_MAP
```
## How to use GUI? 
1. Resume/Pause: click to stop or continue online mission.
2. Stop/Record: click to enter camera path recording mode. Any movement of the camera will be recorded and saved in outputs_gui/saved_paths. You can set the ID of camera path to be recorded by choosing number from "Camera Path" and click reset to delete. Click "Fly" in "Camera Follow Options" to control the camera via keyboard (WASD and direction keys). 
3. Camera Pose: You can save individual camera pose by selecting ID of the camera pose and then clicking "Save". Similarly, click "Load" to move the camera to saved camera poses.
4. History Views: move the camera to planned history viewpoints.
5. 3D Objects: click to visualize 3D objects. You can see different submaps in "Voxel Map". "Mesh" is only available if corresponding mesh is also loaded.
6. Rendering Options: click to show rendering results from Gaussian Splatting map. Only one rendering type among "Depth", "Confidence", "Opacity", "Normal" and "D2N" can be visualized at the same time.
## Evaluation
For rendering evaluation, you need to first generate test views for each scene:
```
python data_generation.py scene=SCENE_NAME
# example:
# python data_generation.py scene=replica/office0
```
This will create a folder containing files of intrinsics and extrinsics of test views in the selected scene.

For mesh evaluation, you need to first extract meshes from the saved GS map:
```
python mesh_generation.py planner=PLANNER_TYPE scene=SCENE_NAME 
# example:
# python mesh_generation.py planner=confidence scene=replica/office0
```
This will generate corresponding mesh files for each GS map saved during the mission.

To get the metrics value:
```
python eval.py planner=PLANNER_TYPE scene=SCENE_NAME test_folder=TEST_FOLDER
# example:
# python eval.py planner=confidence scene=replica/office0 test_folder=dataset/replica/office0
```

We also provide a shell script to run a complete experiment:
```
bash run.sh
```

## Demo
| office0 | office2 | office3 | office4 |
| :-: | :-: | :-: | :-: |
| <video src='https://github.com/user-attachments/assets/feda8281-9a66-4474-a612-866b2723d2a7'> | <video src='https://github.com/user-attachments/assets/523598b0-bd30-40eb-b95b-7d33c4fe69d3'> | <video src='https://github.com/user-attachments/assets/49d8d050-4477-42c0-b69c-263a3ffcbcd3'> | <video src='https://github.com/user-attachments/assets/9c538fbe-da77-4f10-8a80-d66ee13c61a4'> |

| room0 | room1 | room2 | hotel0 |
| :-: | :-: | :-: | :-: |
| <video src='https://github.com/user-attachments/assets/4f0749d0-b4eb-4482-8d99-1b2d070947c1'> | <video src='https://github.com/user-attachments/assets/9be7ca8a-b5c2-4719-b8b9-24eb99dd508a'> | <video src='https://github.com/user-attachments/assets/44beaa5c-c2d3-4995-862e-7749b131ed73'> | <video src='https://github.com/user-attachments/assets/a9a2742c-8966-499a-9b9b-d6c1ae681a2e'> |

## Acknowledgement
Parts of the code are based on [MonoGS](https://github.com/muskie82/MonoGS) and [GaussianSurfels](https://github.com/turandai/gaussian_surfels). We thank the authors for open-sourcing their code.

## Maintainer
Liren Jin, ljin@uni-bonn.de

## Project Funding
This work has been fully funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy, EXC-2070 – 390732324 (PhenoRob).
