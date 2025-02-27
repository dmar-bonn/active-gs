import torch
import time

from utils.operations import *
from utils.common import Camera, Mapper2Gui, FakeQueue, TextColors
from .gaussian_map import GaussianMap
from .voxel_map import VoxelMap


class IncrementalMapper:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

        # map instance
        self.gaussian_map = None
        self.voxel_map = None

        # gui related
        self.use_gui = False
        self.q_mapper2gui = FakeQueue()
        self.q_gui2mapper = FakeQueue()
        self.pause = False
        self.init = False

    @property
    def current_map(self):
        return self.gaussian_map, self.voxel_map

    def load_recorder(self, recorder):
        print("\n ----------load mission recorder----------")
        self.recorder = recorder

    def load_simulator(self, simulator):
        print("\n ----------load simulator----------")
        self.simulator = simulator

    def load_planner(self, planner):
        print("\n ----------load planner----------")
        self.planner = planner

    def init_map(self):
        print("\n ----------initialize map----------")
        self.gaussian_map = GaussianMap(self.cfg.gaussian_map, self.device)
        self.voxel_map = VoxelMap(self.cfg.voxel_map, self.simulator.bbox, self.device)

    def get_new_dataframe(self, i):
        # return way points to the nbv
        path = self.planner.plan(self.current_map, self.simulator, self.recorder)

        # for visualization only
        if self.use_gui:
            for pose in path:
                dataframe = self.simulator.simulate(pose)
                camera_frame = Camera.init_from_mapper(None, dataframe)
                self.q_mapper2gui.put(
                    Mapper2Gui(
                        current_frame=camera_frame,
                    )
                )
                time.sleep(0.05)

        # dataframe at nbv as keyframe
        dataframe = self.simulator.simulate(path[-1])
        camera_frame = Camera.init_from_mapper(i, dataframe)
        self.q_mapper2gui.put(
            Mapper2Gui(
                current_frame=camera_frame,
            )
        )
        return dataframe

    def run(self):
        torch.cuda.empty_cache()
        self.init_map()
        frame_id = 0

        print(
            f"\n {TextColors.MAGENTA}----------Start Active Reconstruction----------{TextColors.RESET}"
        )
        while self.recorder is None or self.recorder.is_alive:
            # pause information from gui
            if not self.q_gui2mapper.empty():
                data_gui2mapper = self.q_gui2mapper.get_nowait()
                self.pause = data_gui2mapper.flag_pause
            if self.pause:
                continue

            print(
                f"\n {TextColors.MAGENTA}----------Step {frame_id+1}----------{TextColors.RESET}"
            )

            print(f"\n {TextColors.GREEN}-----Planning:{TextColors.RESET}")
            dataframe = self.get_new_dataframe(frame_id)
            dataframe = {k: v.to(self.device) for k, v in dataframe.items()}

            print(f"\n {TextColors.GREEN}-----Mapping:{TextColors.RESET}")
            t_mapper_start = time.time()

            # update gaussian map
            self.gaussian_map.update(dataframe)

            # update voxel map
            self.voxel_map.update(dataframe)

            t_mapper = time.time() - t_mapper_start
            frame_id += 1

            # send map to gui for visualization
            self.q_mapper2gui.put(
                Mapper2Gui(
                    gaussians=self.gaussian_map,
                    voxels=self.voxel_map,
                )
            )

            # update recorder or/and save map
            if self.recorder is not None:
                self.recorder.update_time("mapping", t_mapper)
                self.recorder.log()
                self.recorder.save_dataframe(dataframe, f"{frame_id:03}")
                if self.recorder.require_record:
                    self.recorder.save_map(self.gaussian_map, f"{frame_id:03}")
                    self.recorder.save_path()
            time.sleep(0.1)

        print(
            f"\n {TextColors.MAGENTA}----------Finish Reconstruction Mission----------{TextColors.RESET}"
        )
