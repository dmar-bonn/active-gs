import math
import torch.multiprocessing as mp
import os
import imgviz
import torch
from PIL import Image
import open3d as o3d
import pdb
import pickle
import numpy as np


class Camera:
    def __init__(
        self,
        id,
        extrinsic,
        intrinsic=None,
        resolution=None,
        fov=None,
        rgb=None,
        depth=None,
    ):
        self.id = id

        self.extrinsic = extrinsic
        self.intrinsic = intrinsic
        self.rgb = rgb
        self.depth = depth
        if resolution is not None:
            self.H, self.W = resolution
        else:
            self.H = None
            self.W = None
        if self.intrinsic is not None:
            self.fx = self.intrinsic[0, 0]
            self.fy = self.intrinsic[1, 1]
            self.cx = self.intrinsic[0, 2]
            self.cy = self.intrinsic[1, 2]
        else:
            self.fx = None
            self.fy = None
            self.cx = None
            self.cy = None

        if fov is not None:
            self.FoVx, self.FoVy = fov
        else:
            self.FoVx = None
            self.FoVy = None

    @classmethod
    def init_from_mapper(cls, id, frame, with_measurement=True):
        # process rgb image
        if with_measurement:
            rgb = frame["rgb"]
            _, H, W = rgb.shape
            rgb = torch.clamp(rgb, min=0, max=1.0) * 255
            rgb = rgb.byte().permute(1, 2, 0).contiguous().cpu().numpy()
            # rgb = o3d.geometry.Image(rgb)

            # process depth image
            near, far = frame["depth_range"].numpy()
            depth = frame["depth"].squeeze(0).numpy()
            depth = imgviz.depth2rgb(
                depth, min_value=near, max_value=far, colormap="jet"
            )
            depth = torch.from_numpy(depth)
            depth = torch.permute(depth, (2, 0, 1)).float()
            depth = (depth).byte().permute(1, 2, 0).contiguous().cpu().numpy()
            intrinsic = frame["intrinsic"]
            fovx = cls.focal2fov(intrinsic[0, 0], W)
            fovy = cls.focal2fov(intrinsic[1, 1], H)
        else:
            rgb = None
            depth = None
            intrinsic = None
            fovx = None
            fovy = None
            H = None
            W = None

        extrinsic = frame["extrinsic"]
        return cls(
            id,
            extrinsic,
            intrinsic,
            (H, W),
            (fovx, fovy),
            rgb,
            depth,
        )

    @classmethod
    def init_from_gui(cls, id, extrinsic, intrinsic, H, W, fovx, fovy):
        return cls(id, extrinsic, intrinsic, (H, W), (fovx, fovy))

    @staticmethod
    def focal2fov(focal, pixels):
        return 2 * math.atan(pixels / (2 * focal))


class Gui2Mapper:
    flag_pause = False


class GaussianPacket:
    def __init__(self, gaussians):
        self.means = gaussians.get_means.detach().clone()
        self.scales = gaussians.get_scales.detach().clone()
        self.rotations = gaussians.get_rotations.detach().clone()
        self.opacities = gaussians.get_opacities.detach().clone()
        self.harmonics = gaussians.get_harmonics.detach().clone()
        self.confidences = gaussians.get_confidences.clone()
        self.normals = gaussians.get_normals.detach().clone()
        self.background_color = gaussians.background_color.clone()


class VoxelPacket:
    def __init__(self, voxels):
        self.voxel_centers = voxels.voxel_centers.cpu().numpy()
        self.voxel_size = voxels.size.cpu().numpy()
        self.free_mask = voxels.free_mask.cpu().numpy().astype(bool)
        self.occ_mask = voxels.occ_mask.cpu().numpy().astype(bool)
        self.unknown_mask = voxels.unknown_mask.cpu().numpy().astype(bool)
        self.planning_mask = voxels.free_mask_w_margin.cpu().numpy().astype(bool)
        self.unexplored_mask = voxels.unexplored_mask.cpu().numpy().astype(bool)
        self.frontier_mask = voxels.frontier_mask.cpu().numpy().astype(bool)
        self.roi_mask = voxels.roi_mask.cpu().numpy().astype(bool)


class Planner2Gui:
    def __init__(self, view_xyz, view_direction):
        self.view_xyz = view_xyz.numpy()
        self.view_direction = view_direction.numpy()


class Mapper2Gui:
    def __init__(self, gaussians=None, voxels=None, mesh=None, current_frame=None):
        self.has_gaussians = False
        self.has_voxels = False
        self.has_frame = False
        self.has_mesh = False

        if gaussians is not None:
            self.has_gaussians = True
            self.gaussian_packet = GaussianPacket(gaussians)

        if voxels is not None:
            self.has_voxels = True
            self.voxel_packet = VoxelPacket(voxels)

        if mesh is not None:
            self.has_mesh = True
            self.mesh_vertices = np.asarray(mesh.vertices)
            self.mesh_triangles = np.asarray(mesh.faces)

        if current_frame is not None:
            self.has_frame = True
            self.current_frame = current_frame


class FakeQueue:
    def put(self, arg):
        del arg

    def get_nowait(self):
        raise mp.queues.Empty

    def qsize(self):
        return 0

    def empty(self):
        return True


class TextColors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    WHITE = "\033[97m"
    RESET = "\033[0m"  # Reset to default color


class MissionRecorder:
    def __init__(self, save_dir, cfg):
        self.save_dir = save_dir
        self.budget = cfg.budget
        self.record_interval = cfg.record_interval
        self.record_time = cfg.record_interval  # first record time

        self.record_rgbd = cfg.record_rgbd
        self.record_global_path = cfg.record_global_path

        self.time_dict = {"mapping": 0, "planning": 0, "flight": 0}
        self.accum_path_length = 0
        self.camera_params_list = []
        self.global_path_dict = {}
        self.pose_id = 0

    @property
    def is_alive(self):
        return self.t_mission < self.budget

    def save_dataframe(self, dataframe, frame_index):
        extrinsic = dataframe["extrinsic"].cpu().view(-1).numpy().tolist()
        intrinsic = dataframe["intrinsic"].cpu().view(-1).numpy().tolist()
        camera_params = extrinsic + intrinsic
        self.camera_params_list.append(camera_params)

        dataframe_path = os.path.join(self.save_dir, "dataframe")
        if self.record_rgbd:
            rgb_folder = os.path.join(dataframe_path, "rgb")
            os.makedirs(rgb_folder, exist_ok=True)
            depth_folder = os.path.join(dataframe_path, "depth")
            os.makedirs(depth_folder, exist_ok=True)

            rgb = torch.clamp(dataframe["rgb"], min=0, max=1.0) * 255
            rgb = Image.fromarray(
                rgb.byte().permute(1, 2, 0).contiguous().cpu().numpy()
            )
            rgb.save(f"{rgb_folder}/{frame_index}.png")

            near, far = dataframe["depth_range"].cpu().numpy()
            depth = imgviz.depth2rgb(
                dataframe["depth"].squeeze(0).cpu().numpy(),
                min_value=near,
                max_value=far,
                colormap="jet",
            )
            depth = Image.fromarray(depth)
            depth.save(f"{depth_folder}/{frame_index}.png")

            camera_pose_file = os.path.join(dataframe_path, "cameras.txt")
            mode = "a" if os.path.exists(camera_pose_file) else "w"
            with open(camera_pose_file, mode) as f:
                f.write(" ".join(map(str, camera_params)) + "\n")

    def save_map(self, gaussian_map, map_index):
        map_path = os.path.join(self.save_dir, "map")
        os.makedirs(map_path, exist_ok=True)

        print(
            f"\n {TextColors.YELLOW}----------save map after {self.t_mission} seconds----------{TextColors.RESET}"
        )
        gaussian_map.save(map_path, index=map_index)

        # save camera parameters for corresponding gaussian map
        camera_pose_file = os.path.join(map_path, f"cameras_{map_index}.pkl")
        with open(camera_pose_file, "wb") as pickle_file:
            pickle.dump(self.camera_params_list, pickle_file)

        # save mission information
        record_file = f"{map_path}/record_info.txt"
        mode = "a" if os.path.exists(record_file) else "w"
        record_data = [
            map_index,
            self.t_mission,
            self.accum_path_length,
        ]
        with open(record_file, mode) as f:
            f.write(" ".join(map(str, record_data)) + "\n")

    def update_path(self, path, path_length):
        self.accum_path_length += path_length

        if self.record_global_path:
            # non-keyframe
            for camera_pose in path[:-1]:
                camera_data = {"pose": camera_pose, "name": None}
                self.global_path_dict[self.pose_id] = camera_data
                self.pose_id += 1

            camera_data = {"pose": path[-1], "name": self.pose_id}
            self.global_path_dict[self.pose_id] = camera_data
            self.pose_id += 1

    def save_path(self):
        if self.pose_id > 0:
            with open(f"{self.save_dir}/global_path.pkl", "wb") as file:
                pickle.dump(self.global_path_dict, file)
            print("----------save global path----------")
        else:
            print("global path not recorded")

    def update_time(self, item, time_consumption):
        self.time_dict[item] += time_consumption
        print(f"\n {item} time (step): {time_consumption:.2f}")

    def log(self):
        mission_time = self.t_mission
        mapping_percent = self.t_mapping / mission_time
        planning_percent = self.t_planning / mission_time
        flight_percent = self.t_flight / mission_time
        print(f"\n {TextColors.GREEN}-----Log Mission Info:{TextColors.RESET}")
        print(
            f"\n total mission time: {mission_time:.2f},\
                mapping: {mapping_percent *100:.2f}%,\
                planning: {planning_percent*100:.2f}%,\
                flight: {flight_percent*100:.2f}%"
        )
        print(f"\n total travel distance: {self.accum_path_length:.2f}")

    @property
    def require_record(self):
        if self.t_mission > self.record_time:
            self.record_time += self.record_interval
            return True
        else:
            return False

    @property
    def t_mapping(self):
        return self.time_dict["mapping"]

    @property
    def t_planning(self):
        return self.time_dict["planning"]

    @property
    def t_flight(self):
        return self.time_dict["flight"]

    @property
    def t_mission(self):
        return self.t_mapping + self.t_planning + self.t_flight
