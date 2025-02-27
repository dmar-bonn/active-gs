import numpy as np
import torch
from einops import repeat
import time
import pdb
from tqdm import tqdm
import cv2

from .utils import (
    PathPlanner,
    cal_flight_time,
    inplace_rotation,
    rotation_from_z_batch,
    select_points_within_cone,
    wp2path,
)
from utils.common import FakeQueue, Planner2Gui


class PlanBase:
    def __init__(self, cfg, device):
        self.device = device
        self.pitch_angle = cfg.pitch_angle
        self.robot_size = cfg.robot_size
        self.radius = cfg.radius
        self.flight_speed = 1.0
        self.pose = torch.tensor(cfg.init_pose).type(torch.float32)
        self.init = False

        self.path_planner = PathPlanner()
        self.path_length_factor = cfg.path_length_factor

        self.use_confidence = cfg.use_confidence
        self.sample_num = cfg.sample_num
        self.max_roi_sample_num = cfg.max_roi_sample_num  # cfg.roi_sample_num

        # gui related
        self.q_planner2gui = FakeQueue()
        self.q_gui2planner = FakeQueue()

    def plan(self, map, simulator, recorder):
        gaussian_map, voxel_map = map
        t_planning = 0
        if self.init:
            ##### graph update
            t_sampling_start = time.time()
            robot_space = self.get_robot_space(voxel_map)
            voxel_map.update_graph(robot_space)

            ##### view candidate sampling
            if self.max_roi_sample_num > 0:
                voxel_map.update_utility(gaussian_map, self.use_confidence)
                roi_candidates = self.generate_roi_candidates(
                    voxel_map, self.max_roi_sample_num
                )
            else:
                roi_candidates = torch.tensor([])

            if self.sample_num - len(roi_candidates) > 0:
                random_candidates = self.generate_random_candidates(
                    voxel_map, self.sample_num - len(roi_candidates)
                )
            else:
                random_candidates = torch.tensor([])

            total_candidates = torch.cat((roi_candidates, random_candidates), dim=0)
            view_xyz = total_candidates[:, :3, 3]
            view_direction = total_candidates[:, :3, 2]
            t_planning += time.time() - t_sampling_start
            self.q_planner2gui.put(Planner2Gui(view_xyz, view_direction))
            print(
                f"\n generate {len(roi_candidates)} roi samples, {len(random_candidates)} random samples"
            )

            ##### utility calculation
            utility_list, t_utility = self.cal_utility(
                gaussian_map, voxel_map, total_candidates, simulator
            )
            t_planning += t_utility

            ##### path planning
            t_path_start = time.time()
            wp_list, wp_length_list = self.path_planner.search_goal(
                self.pose[:3, 3].numpy(),
                total_candidates[:, :3, 3].numpy(),
                voxel_map,
            )
            t_planning += time.time() - t_path_start

            ##### nbv selection
            score_list = self.cal_view_scores(utility_list, wp_length_list)
            nbv_id = torch.argmax(score_list)
            if nbv_id < len(roi_candidates):
                print("select roi!!!!!!!!!!!!!!!!!!")
            nbv = total_candidates[nbv_id]
            wp_length = wp_length_list[nbv_id]

            # for debugging
            if np.isinf(wp_length):
                print("inf path length!!!!!!!")
                pdb.set_trace()

            wp_indices = wp_list[nbv_id]
            waypoints = voxel_map.index_2_xyz(wp_indices).cpu()

        else:
            # move to closest voxel center as inital position
            nbv = torch.eye(4)
            nbv[:3, :3] = self.pose[:3, :3]
            nbv_index = voxel_map.xyz_2_index(self.pose[:3, 3])
            nbv_xyz = voxel_map.index_2_xyz([nbv_index])[0].cpu()
            nbv[:3, 3] = nbv_xyz
            waypoints = torch.stack([self.pose[:3, 3], nbv_xyz])
            self.init = True

        camera_path, path_length = wp2path(
            self.pose[:3, :3],
            nbv[:3, :3],
            waypoints,
        )
        self.pose = nbv

        if recorder is not None:
            t_flight = cal_flight_time(path_length, flight_speed=self.flight_speed)
            recorder.update_time("planning", t_planning)
            recorder.update_time("flight", t_flight)
            recorder.update_path(camera_path, path_length)

        return camera_path

    def generate_random_candidates(self, voxel_map, num):
        """
        generate random view candidates around current pose
        """

        voxel_centers = voxel_map.voxel_centers.cpu().numpy()
        free_mask = voxel_map.free_mask_w_margin.cpu().numpy()

        range_from_start = np.linalg.norm(
            voxel_centers - self.pose[:3, 3].numpy(), axis=1
        )
        within_range = range_from_start <= self.radius
        valid_mask = free_mask & within_range
        valid_centers = voxel_centers[valid_mask]
        random_indices = np.random.choice(len(valid_centers), size=num)
        view_positions = valid_centers[random_indices]
        candidates = inplace_rotation(
            view_positions, pitch_angle=self.pitch_angle, num=num
        )
        return candidates

    def generate_roi_candidates(self, voxel_map, num):
        """
        generate targeted view candidates arount ROI
        """

        roi_candiates = torch.tensor([])
        sample_per_roi = 5
        free_mask = voxel_map.free_mask_w_margin
        free_points = voxel_map.voxel_centers[free_mask]

        roi_mask = voxel_map.roi_mask
        roi_centers = voxel_map.voxel_centers[roi_mask]
        roi_normals = voxel_map.voxel_normal[roi_mask]
        roi_distance = torch.linalg.norm(
            roi_centers - self.pose[:3, 3].unsqueeze(0).to(self.device), dim=1
        )
        _, closest_roi_index = torch.sort(roi_distance)
        for roi_index in closest_roi_index:
            roi_center = roi_centers[roi_index]
            roi_normal = roi_normals[roi_index]
            candiate_positions, candidate_views = select_points_within_cone(
                roi_center,
                roi_normal,
                d_close=0.3,
                d_far=2.0,
                cosine_sim=0.5,
                free_points=free_points,
                voxel_map=voxel_map,
                pitch_angle=self.pitch_angle,
            )
            num_candidates = len(candiate_positions)

            # assign voxel center as final xyz
            if num_candidates > 0:
                if num_candidates > sample_per_roi:
                    selected_index = np.random.choice(
                        range(num_candidates),
                        size=sample_per_roi,
                        replace=False,
                    )
                    candiate_positions = candiate_positions[selected_index]
                    candidate_views = candidate_views[selected_index]

                Ts = torch.tensor(
                    repeat(np.eye(4), "h w -> n h w", n=len(candiate_positions))
                )
                Ts[:, :3, 3] = candiate_positions
                Ts[:, :3, :3] = rotation_from_z_batch(candidate_views)

                roi_candiates = torch.cat((roi_candiates, Ts), dim=0)

            if len(roi_candiates) >= num:
                return roi_candiates.type(torch.float32).cpu()

        return roi_candiates.type(torch.float32).cpu()

    def get_robot_space(self, voxel_map):
        range_from_start = torch.linalg.norm(
            voxel_map.voxel_centers - self.pose[:3, 3].unsqueeze(0).to(self.device),
            dim=1,
        )
        robot_space = range_from_start < self.robot_size
        return robot_space

    def cal_view_scores(self, view_utilities, path_lengths):
        """
        calculate the score of each viewpoint based on its utility and travel cost
        """

        path_lengths = torch.tensor(path_lengths)
        valid_candidate_mask = ~torch.isinf(path_lengths)

        path_lengths = path_lengths / torch.sum(path_lengths[valid_candidate_mask])
        path_lengths[~valid_candidate_mask] = 10000000

        view_utilities = view_utilities / torch.sum(view_utilities)
        view_utilities[torch.isnan(view_utilities)] = 0
        if torch.all(view_utilities == 0):
            view_scores = torch.rand_like(view_utilities)
        else:
            view_scores = view_utilities - self.path_length_factor * path_lengths
        return view_scores

    def cal_utility(self):
        raise NotImplementedError
