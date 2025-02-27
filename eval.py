import hydra
import torch
import warnings
import os
import json
import open3d as o3d
import numpy as np

from utils.evaluation_tool import EvaluationTool
from mapping.gaussian_map import GaussianMap
from simulator import get_simulator


warnings.simplefilter("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(
    version_base=None,
    config_path="./config",
    config_name="eval",
)
def main(cfg):
    experiment_path = os.path.join(
        cfg.experiment.output_dir,
        str(cfg.experiment.exp_id),
        cfg.scene.scene_name,
        cfg.planner.planner_name,
        str(cfg.experiment.run_id),
    )
    eval_mode = cfg.eval_mode

    record_info_file = f"{experiment_path}/map/record_info.txt"
    if os.path.exists(record_info_file):
        record_info = np.loadtxt(record_info_file)
    else:
        print("no record file!!!")
        return False
    id_list = record_info[:, 0]
    timer_list = record_info[:, 1]
    path_list = record_info[:, 2]
    map_list = []
    mesh_list = []

    for map_id in id_list:
        map_id = int(map_id)
        map_file = f"{experiment_path}/map/map_{map_id:03}.th"
        os.path.exists(map_file)
        gaussain_map = GaussianMap(None, device)
        gaussain_map.load(map_file)
        map_list.append(gaussain_map)

        mesh_file = f"{experiment_path}/map/mesh_{map_id:03}.ply"
        os.path.exists(mesh_file)
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        mesh_list.append(mesh)

    simulator = get_simulator(cfg)

    print("\n----------start evaluation----------")
    eval_tool = EvaluationTool(
        map_list,
        mesh_list,
        cfg.test_folder,
        experiment_path,
        eval_mode,
        device,
        simulator=simulator,
    )
    eval_result = eval_tool.eval()
    eval_result["step"] = list(id_list)
    eval_result["time"] = list(timer_list)
    eval_result["path_length"] = list(path_list)

    result_file = os.path.join(experiment_path, "final_result.json")

    if os.path.exists(result_file):
        with open(result_file, "r") as f:
            result_data = json.load(f)
        result_data.update(eval_result)
    else:
        result_data = eval_result
    with open(result_file, "w") as f:
        json.dump(
            result_data,
            f,
            indent=4,
        )


if __name__ == "__main__":
    main()
