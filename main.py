import hydra
import torch
import warnings
import torch.multiprocessing as mp
import os
import yaml
from omegaconf import OmegaConf

from visualization import gui
from utils.common import MissionRecorder
from simulator import get_simulator
from mapping import get_mapper
from planning import get_planner


warnings.simplefilter("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(
    version_base=None,
    config_path="./config",
    config_name="main",
)
def main(cfg):
    # set up mode config
    if cfg.debug:
        mission_recorder = None

    else:
        experiment_path = os.path.join(
            cfg.experiment.output_dir,
            str(cfg.experiment.exp_id),
            cfg.scene.scene_name,
            cfg.planner.planner_name,
            str(cfg.experiment.run_id),
        )
        os.makedirs(experiment_path, exist_ok=True)

        # save experiment configuration
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        with open(f"{experiment_path}/exp_config.yaml", "w") as file:
            yaml.dump(cfg_dict, file)

        mission_recorder = MissionRecorder(experiment_path, cfg.experiment)

    # load components
    mapping_agent = get_mapper(cfg, device)
    simulator = get_simulator(cfg)
    planner = get_planner(cfg, device)

    # set up gui messages
    mp.set_start_method("spawn")
    if cfg.use_gui:
        init_event = mp.Event()
        q_mapper2gui = mp.Queue()
        q_gui2mapper = mp.Queue()
        q_planner2gui = mp.Queue()
        q_gui2planner = mp.Queue()

        mapping_agent.use_gui = True
        mapping_agent.q_mapper2gui = q_mapper2gui
        mapping_agent.q_gui2mapper = q_gui2mapper

        planner.q_planner2gui = q_planner2gui
        planner.q_gui2planner = q_planner2gui

        params_gui = {
            "mapper_receive": q_mapper2gui,
            "mapper_send": q_gui2mapper,
            "planner_receive": q_planner2gui,
            "planner_send": q_gui2planner,
        }
        gui_process = mp.Process(
            target=gui.run,
            args=(init_event, cfg.gui, params_gui),
        )
        gui_process.start()
        init_event.wait()

    # load components to mapping module
    mapping_agent.load_recorder(mission_recorder)
    mapping_agent.load_simulator(simulator)
    mapping_agent.load_planner(planner)

    # start mission
    mapping_agent.run()

    # wait for gui to be closed
    if cfg.use_gui:
        gui_process.join()


if __name__ == "__main__":
    main()
