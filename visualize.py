import torch
import warnings
import click
import time
import pickle
import trimesh
import torch.multiprocessing as mp

from mapping.gaussian_map import GaussianMap
from visualization import gui
from utils.common import Mapper2Gui, Camera

warnings.simplefilter("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@click.command()
@click.option("--gaussian_file", "-G", type=str, help="gaussian map file")
@click.option("--mesh_file", "-M", type=str, help="mesh file")
@click.option("--camera_path_file", "-C", type=str, help="camera path file")
def main(gaussian_file=None, mesh_file=None, camera_path_file=None):
    # start gui process
    mp.set_start_method("spawn")
    init_event = mp.Event()
    q_mapper2gui = mp.Queue()
    params_gui = {
        "mapper_receive": q_mapper2gui,
    }
    gui_process = mp.Process(
        target=gui.run,
        args=(init_event, None, params_gui),
    )
    gui_process.start()
    init_event.wait()

    # load gaussian map
    if gaussian_file is not None:
        print("----------loading gaussian map----------")
        gaussian_map = GaussianMap(None, device)
        gaussian_map.load(gaussian_file)
        q_mapper2gui.put(
            Mapper2Gui(
                gaussians=gaussian_map,
            )
        )
        time.sleep(0.5)

    if mesh_file is not None:
        print("----------loading mesh----------")
        mesh = trimesh.load_mesh(mesh_file)
        q_mapper2gui.put(
            Mapper2Gui(
                mesh=mesh,
            )
        )
        time.sleep(0.5)

    # load camera path
    if camera_path_file is not None:
        print("----------loading camera path----------")
        try:
            with open(camera_path_file, "rb") as file:
                camera_path_dict = pickle.load(file)
        except Exception as e:
            print(f"An error occurred: {e}")
        for id, camera_item in camera_path_dict.items():
            camera_pose = camera_item["pose"]
            frame_name = camera_item["name"]
            dataframe = {
                "extrinsic": camera_pose,
            }
            camera_frame = Camera.init_from_mapper(
                frame_name, dataframe, with_measurement=False
            )
            q_mapper2gui.put(
                Mapper2Gui(
                    current_frame=camera_frame,
                )
            )
            time.sleep(0.05)
        print("----------camera path complete!!!!----------")

    gui_process.join()


if __name__ == "__main__":
    main()
