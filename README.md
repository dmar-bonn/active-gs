# ActiveGS: Active Scene Reconstruction Using Gaussian Splatting

Liren Jin<sup>1</sup>, Xingguang Zhong<sup>1</sup>, Yue Pan<sup>1</sup>, Jens Behley<sup>1</sup>, Cyrill Stachniss<sup>1</sup>, Marija Popovic<sup>2</sup> 

<sup>1</sup> University of Bonn, <sup>2</sup> TU Delft

<a target="_blank" href="https://arxiv.org/abs/2412.17769">
    <img src="https://img.shields.io/badge/arXiv-2412.17769-b31b1b.svg" alt="arXiv Paper">
</a>


![teaser](https://github.com/user-attachments/assets/f7c16143-fc3c-4d69-88f3-f7bd43d9de85)
## Setup
We test the following setup on Ubuntu20 with CUDA11.3

Clone ActiveGS repo:
```
git clone git@github.com:dmar-bonn/active-gs.git
cd active-gs
```

Create environment:
```
bash envs/build.sh
```

Download Replica dataset:
```
bash data/replica_download.sh
conda activate active-gs
```
## Run
Run online mission:
```
python main.py planner=PLANNER_TYPE scene=SCENE_NAME
# example: 
# python main.py planner=confidence scene=replica/office0 use_gui=true
```
If use_gui is set to true, you should be able to see a GUI running.


For visualizing the built GS map:
```
python visualize.py -G PATH_TO_GS_MAP
```
## How to use GUI? 
TODO
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
./run.sh
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
