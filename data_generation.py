import hydra
import torch
import time
import warnings
import torch.multiprocessing as mp
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from einops import repeat

from utils.common import Mapper2Gui, Camera
from utils.operations import random_rotation
from visualization import gui
from simulator import get_simulator
from planning import get_planner
from mapping.voxel_map import VoxelMap


warnings.simplefilter("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(
    version_base=None,
    config_path="./config",
    config_name="data_generation",
)
def main(cfg):
    simulator = get_simulator(cfg)
    save_path = os.path.join(cfg.dataset_path, simulator.scene_name)
    planner = get_planner(cfg, device)
    voxel_map = VoxelMap(cfg.mapper, simulator.bbox, device)
    iter = 0
    converged = 0

    # set up gui messages
    mp.set_start_method("spawn")
    if cfg.use_gui:
        init_event = mp.Event()
        q_mapper2gui = mp.Queue()
        q_gui2mapper = mp.Queue()
        params_gui = {
            "mapper_receive": q_mapper2gui,
            "mapper_send": q_gui2mapper,
        }
        gui_process = mp.Process(
            target=gui.run,
            args=(
                init_event,
                cfg.gui,
                params_gui,
            ),
        )
        gui_process.start()
        init_event.wait()

    # map free space in the scene
    while iter < cfg.max_iter and converged < cfg.converged_step:
        path = planner.plan([None, voxel_map], simulator, None)
        pose = path[-1]
        dataframe = simulator.simulate(torch.tensor(pose), require_gt=True)
        camera_frame = Camera.init_from_mapper(iter, dataframe)

        dataframe = {k: v.to(device) for k, v in dataframe.items()}
        voxel_state_old = voxel_map.unexplored_mask.clone()
        voxel_map.update(dataframe)
        voxel_state_new = voxel_map.unexplored_mask.clone()
        changes = track_changes(voxel_state_old, voxel_state_new)
        if changes == 0:
            converged += 1
        else:
            converged = 0
        iter += 1
        if cfg.use_gui:
            q_mapper2gui.put(
                Mapper2Gui(
                    current_frame=camera_frame,
                    gaussians=None,
                    voxels=voxel_map,
                )
            )
            time.sleep(0.5)

    test_views = generate_test_views(voxel_map, cfg.num_views)
    record_data(save_path, simulator, test_views, cfg.save_pose_only)


def record_data(path, simulator, test_views, save_pose_only):
    print(f"\n ---------- generating {len(test_views)} test views ----------")

    os.makedirs(path, exist_ok=True)

    if not save_pose_only:
        os.makedirs(f"{path}/rgb", exist_ok=True)
        os.makedirs(f"{path}/depth", exist_ok=True)

        for i, pose in tqdm(enumerate(test_views), total=len(test_views)):
            dataframe = simulator.simulate(torch.tensor(pose))
            rgb = dataframe["rgb"]
            depth = dataframe["depth"]

            rgb_img = Image.fromarray(
                (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8), mode="RGB"
            )
            rgb_img.save(os.path.join(path, "rgb/{:05}.png".format(i)))
            depth_img = Image.fromarray(
                (depth.squeeze(0).numpy() / 10 * 255).astype(np.uint8), mode="L"
            )
            depth_img.save(os.path.join(path, "depth/{:05}.png".format(i)))

    np.savetxt(f"{path}/traj.txt", np.array(test_views.reshape(len(test_views), -1)))
    np.savetxt(f"{path}/intrinsic.txt", simulator.intrinsic.numpy().reshape(-1))


def track_changes(old_state, new_state):
    unknown_old = old_state == 0
    unknown_new = new_state == 0
    change_mask = unknown_old != unknown_new
    changes = torch.sum(change_mask)
    return changes


def generate_test_views(voxel_map, num_views):
    voxel_center = voxel_map.voxel_centers.cpu().numpy()
    # voxel_states = voxel_map.voxel_states
    voxel_size = voxel_map.size.cpu().numpy()

    free_mask = voxel_map.free_mask.cpu().numpy()
    num_free_voxel = np.sum(free_mask)
    points_needed = max(num_views, num_free_voxel)

    num_per_voxel = np.ceil(points_needed / num_free_voxel).astype(int)

    voxel_min = voxel_center[free_mask] - 0.5 * voxel_size
    voxel_max = voxel_min + voxel_size

    repeated_voxel_min = np.repeat(voxel_min, num_per_voxel, axis=0)
    repeated_voxel_max = np.repeat(voxel_max, num_per_voxel, axis=0)

    points = np.random.uniform(
        repeated_voxel_min, repeated_voxel_max, size=(len(repeated_voxel_min), 3)
    )
    if len(points) > num_views:
        indices = np.random.choice(len(points), num_views, replace=False)
        sampled_points = points[indices]
    else:
        sampled_points = points

    Ts = repeat(np.eye(4), "h w -> n h w", n=num_views)
    Ts[:, :3, 3] = sampled_points
    Ts[:, :3, :3] = random_rotation(num_views, pitch_angle=None)
    return torch.tensor(Ts).type(torch.float32)


if __name__ == "__main__":
    main()
