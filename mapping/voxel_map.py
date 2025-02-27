import numpy as np
import torch
from collections import defaultdict
from scipy.ndimage import binary_dilation, generate_binary_structure

from utils.operations import *
from utils.common import TextColors
from .utils import cal_distance


class VoxelMap:
    def __init__(self, cfg, bbox, device):
        self.device = device
        self.min_gaussian_per_voxel = cfg.min_gaussian_per_voxel
        bbox = torch.tensor(bbox)
        extents = bbox[1] - bbox[0]
        map_resolution = torch.tensor(cfg.map_resolution)
        dim = torch.ceil(extents / map_resolution).int()
        size = extents / dim
        self.occ_structure_element = self._create_spherical_structuring_element(
            np.max(np.array(cfg.safety_margin) / size.numpy())
        )
        self.frontier_structure_element = generate_binary_structure(3, 1)

        indices_x = torch.arange(dim[0])
        indices_y = torch.arange(dim[1])
        indices_z = torch.arange(dim[2])
        grid_x, grid_y, grid_z = torch.meshgrid(
            indices_x, indices_y, indices_z, indexing="ij"
        )
        centers_x = bbox[0][0] + (grid_x + 0.5) * size[0]
        centers_y = bbox[0][1] + (grid_y + 0.5) * size[1]
        centers_z = bbox[0][2] + (grid_z + 0.5) * size[2]

        self.voxel_centers = torch.stack(
            (centers_x, centers_y, centers_z), dim=-1
        ).view(-1, 3)
        self.voxel_indices = torch.floor((self.voxel_centers - bbox[0]) / size).int()
        self.points_3d_hom = torch.cat(
            (
                self.voxel_centers,
                torch.ones((self.voxel_centers.shape[0], 1)),
            ),
            dim=-1,
        )  # (N, 4)

        self.occ_thres = 0.8
        self.free_thres = 0.2
        self.voxel_lo = torch.zeros(torch.prod(dim))  # init prob = 0.5

        self.unexplored_mask = torch.ones(torch.prod(dim), dtype=torch.bool)
        self.planning_mask = torch.zeros(torch.prod(dim))
        self.roi_mask = torch.zeros(torch.prod(dim), dtype=torch.bool)

        self.graph = VoxelGrpah(size.numpy(), dim.numpy(), self.voxel_indices.numpy())

        self.dim = dim
        self.size = size
        self.bbox = bbox
        self.to_device(device)

    def update_utility(self, gaussian_map, use_confidence, confidence_thres=0.3):
        """
        find roi voxels: frontier and low-confidence
        """

        self.voxel_normal = torch.zeros((torch.prod(self.dim), 3), device=self.device)
        raw_roi_mask = self.frontier_mask

        if use_confidence:
            gaussian_mean = gaussian_map.get_means.detach()
            gaussian_normal = gaussian_map.get_normals.detach()
            gaussian_confidence = gaussian_map.get_confidences.detach()
            gaussian_opacity = gaussian_map.get_opacities.detach()

            voxel_indices, valid_mask = self._voxelize(gaussian_mean)
            voxel_indices = voxel_indices[valid_mask]
            gaussian_mean = gaussian_mean[valid_mask]
            gaussian_normal = gaussian_normal[valid_mask]
            gaussian_confidence = gaussian_confidence[valid_mask]
            gaussian_opacity = gaussian_opacity[valid_mask]

            # get high-opacity low-confidence gaussians
            confidence_mask = gaussian_confidence < confidence_thres
            confidence_mask *= gaussian_opacity > 0.7

            voxel_indices = voxel_indices[confidence_mask]
            gaussian_mean = gaussian_mean[confidence_mask]
            gaussian_normal = gaussian_normal[confidence_mask]

            linear_indices = self.to_linear_indices(voxel_indices)
            voxel_sum = torch.zeros(
                torch.prod(self.dim), dtype=torch.int64, device=gaussian_mean.device
            )
            voxel_sum = voxel_sum.scatter_add(
                0, linear_indices, torch.ones_like(linear_indices, dtype=torch.int64)
            )

            voxel_normal_sum = torch.zeros(
                (torch.prod(self.dim), 3), device=gaussian_mean.device
            )
            voxel_normal_sum = voxel_normal_sum.scatter_add(
                0,
                linear_indices.unsqueeze(1).expand(-1, 3),
                gaussian_normal,
            )

            # add voxels with low-confidence gaussians to roi set
            update_mask = voxel_sum > self.min_gaussian_per_voxel
            mean_normals = voxel_normal_sum / voxel_sum.unsqueeze(1)
            self.voxel_normal[update_mask] = torch.nn.functional.normalize(
                mean_normals[update_mask], dim=-1
            )
            raw_roi_mask += update_mask

        self.roi_mask = self.get_roi_mask(raw_roi_mask)

    def update_graph(self, robot_space):
        """
        update graph for path planning
        """

        planning_mask = self.free_mask_w_margin + robot_space
        self.graph.update_graph(planning_mask.cpu().numpy())

    def update(self, dataframe):
        """
        update voxel map state given posed depth observation
        """

        print(f" {TextColors.CYAN}Update Voxel Map{TextColors.RESET}")
        depth_map = dataframe["depth"].squeeze(0)
        extrinsic = dataframe["extrinsic"]
        intrinsic = dataframe["intrinsic"]
        depth_range = dataframe["depth_range"]
        H, W = depth_map.shape
        out_range_mask = depth_map == -1.0
        depth_map_clone = depth_map.clone()
        depth_map_clone[out_range_mask] = depth_range[1]

        points_2d, points_depth = self._project_3d_points(extrinsic, intrinsic)
        frustum_pass_mask, _ = self._get_frustum_mask(
            points_2d, points_depth, depth_map_clone
        )
        xy_ray, _ = sample_image_grid((H, W), device=self.device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        origins, directions = get_world_rays(xy_ray, extrinsic, intrinsic)
        invalid_depth_mask = depth_map.view(-1) < 0.0  # out of range or missing surface
        points_3d = (origins + directions * depth_map.view(-1, 1, 1)).view(H * W, 3)
        points_3d = points_3d - self.bbox[0]
        indices = torch.floor(points_3d / self.size).int()
        valid_index_mask = (
            torch.all(indices >= 0, dim=1)
            & torch.all(indices < self.dim, dim=1)
            & (~invalid_depth_mask)
        )
        valid_indices = indices[valid_index_mask]
        x_indices = valid_indices[:, 0]
        y_indices = valid_indices[:, 1]
        z_indices = valid_indices[:, 2]
        frustum_hit_mask = torch.zeros(*self.dim, dtype=torch.bool, device=self.device)
        frustum_hit_mask[x_indices, y_indices, z_indices] = True
        frustum_hit_mask = frustum_hit_mask.view(-1)

        # remove occ voxel from free mask
        frustum_pass_mask &= ~frustum_hit_mask
        dist_pass = cal_distance(
            self.voxel_centers[frustum_pass_mask], extrinsic[:3, 3]
        )
        weighted_pass_lo = 2.8 * self.inverse_sensor_model(dist_pass)
        dist_hit = cal_distance(self.voxel_centers[frustum_hit_mask], extrinsic[:3, 3])
        weighted_hit_lo = 2.8 * self.inverse_sensor_model(dist_hit)

        # update log odds
        self.voxel_lo[frustum_hit_mask] += weighted_hit_lo  # prob += 0.4
        self.voxel_lo[frustum_pass_mask] -= weighted_pass_lo  # prob -= 0.4
        self.voxel_lo = torch.clip(
            self.voxel_lo, min=-4.5, max=4.5
        )  # prob in (0.01 , 0.99)

        self.unexplored_mask[frustum_hit_mask] = False
        self.unexplored_mask[frustum_pass_mask] = False

    def to_linear_indices(self, voxel_indices):
        """
        convert ijk index to linear index
        """

        linear_indices = (
            voxel_indices[:, 0] * (self.dim[1] * self.dim[2])
            + voxel_indices[:, 1] * self.dim[2]
            + voxel_indices[:, 2]
        ).to(torch.int64)
        return linear_indices

    def _voxelize(self, positions):
        """
        find ijk index given xyz
        """

        relative_positions = positions - self.bbox[0]
        voxel_indices = torch.floor(relative_positions / self.size).int()
        valid_mask = torch.all(voxel_indices >= 0, dim=1) & torch.all(
            voxel_indices < self.dim, dim=1
        )
        return voxel_indices, valid_mask

    def _dilate_mask(self, mask, structure_element):
        dilated_mask = binary_dilation(
            mask,
            structure=structure_element,
        )
        return dilated_mask

    @staticmethod
    def _create_spherical_structuring_element(radius):
        """
        create a spherical structuring element with a given radius.
        """

        L = np.arange(-radius, radius + 1)
        X, Y, Z = np.meshgrid(L, L, L)
        structuring_element = (X**2 + Y**2 + Z**2) <= radius**2
        return structuring_element

    def _project_3d_points(self, extrinsic, intrinsic):
        """
        project 3d points on image plane
        """

        points_camera_hom = (extrinsic.inverse() @ self.points_3d_hom.T).T
        points_camera = points_camera_hom[:, :3]  # (N ,3)
        points_depth = points_camera[:, 2]  # (N)
        points_image_hom = (intrinsic @ points_camera.T).T
        points_image = points_image_hom[:, :2] / points_image_hom[:, 2].unsqueeze(
            -1
        )  # (N, 2)
        return points_image, points_depth

    def _get_frustum_mask(self, points_2d, points_depth, depth_map):
        """
        get visible points within view frustum
        """

        h, w = depth_map.shape
        front_mask = points_depth > 0  # (N)
        points_2d[:, 0] = points_2d[:, 0] * w
        points_2d[:, 1] = points_2d[:, 1] * h
        pixels_2d = torch.round(points_2d)
        N = pixels_2d.shape[0]
        x_indices = points_2d[:, 0]
        y_indices = points_2d[:, 1]
        valid_x = (x_indices >= 0) & (x_indices < w)
        valid_y = (y_indices >= 0) & (y_indices < h)
        valid_coords = valid_x & valid_y

        valid_indices = valid_coords.nonzero(as_tuple=True)[0]
        valid_x_coords = x_indices[valid_coords].long()
        valid_y_coords = y_indices[valid_coords].long()

        depth_values = torch.full((N,), -1.0, device=depth_map.device)
        depth_values[valid_indices] = depth_map[valid_y_coords, valid_x_coords]
        invalid_depth_mask = depth_values < 0.0  # invalid depth measurements
        depth_mask = depth_values > points_depth

        fov_mask = front_mask & valid_x & valid_y
        frustum_mask = fov_mask & depth_mask
        invalid_mask = fov_mask & invalid_depth_mask
        return frustum_mask, invalid_mask

    def cal_visible_mask(self, extrinsic, intrinsic, depth):
        """
        get mask for all visible voxels at given view with its depth
        """

        points_2d, points_depth = self._project_3d_points(extrinsic, intrinsic)
        frustum_mask, _ = self._get_frustum_mask(points_2d, points_depth, depth)
        return frustum_mask

    def get_roi_mask(self, raw_roi_mask):
        """
        remove roi without free neighbors
        """

        dilated_free_mask = self._dilate_mask(
            self.free_mask.view(*self.dim).clone().cpu().numpy(),
            self.frontier_structure_element,
        )
        dilated_free_mask = torch.tensor(
            dilated_free_mask, dtype=torch.bool, device=self.device
        ).view(-1)
        return raw_roi_mask & dilated_free_mask

    def check_visible_direction(self, point):
        voxel_index = torch.tensor(self.xyz_2_index(point), device=self.device)
        directions = torch.tensor(
            [[x, y, z] for x in [-1, 0, 1] for y in [-1, 0, 1] for z in [-1, 0, 1]],
            device=self.device,
        )
        directions = directions[torch.any(directions != 0, dim=1)]
        neighbor_indices = voxel_index.unsqueeze(0) + directions

        in_bounds = torch.all(neighbor_indices >= 0, dim=1) & torch.all(
            neighbor_indices < self.dim, dim=1
        )
        neighbor_indices = neighbor_indices[in_bounds]
        free_neighbor_mask = self.free_mask.view(*self.dim)[
            neighbor_indices[:, 0], neighbor_indices[:, 1], neighbor_indices[:, 2]
        ]
        if torch.sum(free_neighbor_mask) > 0:
            free_neighbor_indices = neighbor_indices[free_neighbor_mask]
            free_neighbor_indices = free_neighbor_indices.view(-1, 3)
            free_neighbor_xyzs = self.index_2_xyz(free_neighbor_indices)
            view_directions = free_neighbor_xyzs - point.unsqueeze(0)
            view_directions = view_directions / view_directions.norm(
                dim=-1, keepdim=True
            )
            view_direction = torch.mean(view_directions, dim=0)
            return view_direction

        else:
            return None

    @property
    def voxel_states(self):
        return self.inverse_log_odds(self.voxel_lo)

    @property
    def free_mask_w_margin(self):
        dilated_occ_mask = self._dilate_mask(
            self.occ_mask.view(*self.dim).clone().cpu().numpy(),
            self.occ_structure_element,
        )
        dilated_occ_mask = torch.tensor(
            dilated_occ_mask, dtype=torch.bool, device=self.device
        ).view(-1)

        return self.free_mask & ~dilated_occ_mask

    @property
    def frontier_mask(self):
        dilated_unexplored_mask = self._dilate_mask(
            self.unexplored_mask.view(*self.dim).clone().cpu().numpy(),
            self.frontier_structure_element,
        )
        dilated_unexplored_mask = torch.tensor(
            dilated_unexplored_mask, dtype=torch.bool, device=self.device
        ).view(-1)
        return dilated_unexplored_mask & self.free_mask

    @property
    def free_mask(self):
        return self.voxel_states <= self.free_thres

    @property
    def occ_mask(self):
        return self.voxel_states >= self.occ_thres

    @property
    def unknown_mask(self):
        return (~self.free_mask) & (~self.occ_mask)

    def index_2_xyz(self, indices):
        indices = torch.tensor(indices, device=self.device).view(-1, 3)
        indices_1d = (
            indices[:, 0] * self.dim[1] * self.dim[2]
            + indices[:, 1] * self.dim[2]
            + indices[:, 2]
        )
        positions = self.voxel_centers[indices_1d]
        return positions

    def xyz_2_index(self, xyz):
        xyz = xyz.to(self.device)
        relative_positions = xyz - self.bbox[0]
        voxel_indices = torch.floor(relative_positions / self.size).int()
        return voxel_indices.tolist()

    def in_free_space(self, positions):
        """
        check whether point are in free space
        """

        final_mask = torch.zeros(len(positions), dtype=torch.bool, device=self.device)
        indices, valid_index_mask = self._voxelize(positions)
        valid_indices = indices[valid_index_mask]
        x_indices = valid_indices[:, 0]
        y_indices = valid_indices[:, 1]
        z_indices = valid_indices[:, 2]
        query_states = self.free_mask_w_margin.view(*self.dim)[
            x_indices, y_indices, z_indices
        ]
        final_mask[valid_index_mask] = query_states
        return final_mask

    def voxel_filter(self, points):
        """
        check whether points are within occupied space
        """

        x_min, y_min, z_min = self.bbox[0] - 0.05  # bbox with margin
        x_max, y_max, z_max = self.bbox[1] + 0.05
        x_mask = (points[:, 0] > x_min) & (points[:, 0] < x_max)
        y_mask = (points[:, 1] > y_min) & (points[:, 1] < y_max)
        z_mask = (points[:, 2] > z_min) & (points[:, 2] < z_max)
        in_box_mask = x_mask & y_mask & z_mask
        in_free_space_mask = self.in_free_space(points)
        final_mask = in_box_mask * (~in_free_space_mask)
        return final_mask

    @staticmethod
    def log_odds(p):
        """convert probability to log-odds"""

        return torch.log(p / (1 - p))

    @staticmethod
    def inverse_log_odds(l):
        """convert log-odds to probability"""

        return 1 - 1 / (1 + torch.exp(l))

    @staticmethod
    def inverse_sensor_model(distance):
        weight = torch.clip(1 - 0.1 * distance, min=0.0, max=1.0)
        return weight

    def to_device(self, device):
        self.voxel_centers = self.voxel_centers.to(device)
        self.voxel_indices = self.voxel_indices.to(device)
        self.points_3d_hom = self.points_3d_hom.to(device)
        self.voxel_lo = self.voxel_lo.to(device)
        self.unexplored_mask = self.unexplored_mask.to(device)
        self.roi_mask = self.roi_mask.to(device)
        self.dim = self.dim.to(device)
        self.size = self.size.to(device)
        self.bbox = self.bbox.to(device)

    def save(self, save_path, index="final"):
        map_state = {
            "voxel_center": self.voxel_centers,
            "voxel_lo": self.voxel_lo,
            "unexplored_mask": self.unexplored_mask,
            "dim": self.dim,
            "size": self.size,
        }
        torch.save(map_state, f"{save_path}/voxel_map_{index}.th")


class VoxelGrpah:
    def __init__(self, voxel_size, voxel_dim, voxel_indices):
        offsets = [-1, 0, 1]
        directions = np.array(
            [[x, y, z] for x in offsets for y in offsets for z in offsets]
        )
        self.directions = directions[np.any(directions != 0, axis=1)]
        self.direction_distances = np.linalg.norm(self.directions * voxel_size, axis=1)
        self.dim = voxel_dim
        self.indices = voxel_indices
        self.previous_traversable_mask = None
        self.dense_graph = defaultdict(list)

    def update_graph(self, current_traversable_mask):
        """
        update graph based on current free space in the voxel map
        """

        current_traversable_mask = current_traversable_mask.reshape(self.dim)
        if self.previous_traversable_mask is None:
            to_free_indices = np.argwhere(current_traversable_mask)
            self._add_edges_bulk(to_free_indices, current_traversable_mask)
        else:
            to_free_mask = ~self.previous_traversable_mask & current_traversable_mask
            to_occupied_mask = (
                self.previous_traversable_mask & ~current_traversable_mask
            )
            to_free_indices = np.argwhere(to_free_mask)
            to_occupied_indices = np.argwhere(to_occupied_mask)

            # add edges in bulk for to-free transitions
            self._add_edges_bulk(to_free_indices, current_traversable_mask)
            # remove edges in bulk for to-occupied transitions
            self._remove_edges_bulk(to_occupied_indices)

        self.previous_traversable_mask = current_traversable_mask

    def _add_edges_bulk(self, center_indices, valid_mask):
        """
        add edges for all voxels that became free
        """

        for center_index in center_indices:
            # compute neighbors for this voxel, shape will be (n, 3)
            neighbor_indices = center_index + self.directions

            # check which neighbors are in bounds, results in a (n,) boolean array
            in_bounds = np.all(neighbor_indices >= 0, axis=1) & np.all(
                neighbor_indices < self.dim, axis=1
            )

            # apply the in-bounds mask to filter out-of-bounds neighbors
            neighbor_indices = neighbor_indices[in_bounds]

            # reshape the valid mask into a 3D form and check which neighbors are free
            free_neighbor_mask = valid_mask[
                neighbor_indices[:, 0],
                neighbor_indices[:, 1],
                neighbor_indices[:, 2],
            ]

            # filter out valid free neighbors
            free_neighbor_indices = neighbor_indices[free_neighbor_mask]

            # get the direction distances corresponding to valid in-bound neighbors
            valid_directions_dist = self.direction_distances[in_bounds][
                free_neighbor_mask
            ]

            # update the graph for this center voxel
            if len(free_neighbor_indices) > 0:
                center_index_tuple = tuple(center_index)
                self.dense_graph[center_index_tuple] = [
                    (tuple(neighbor), dist)
                    for neighbor, dist in zip(
                        free_neighbor_indices, valid_directions_dist
                    )
                ]
                # add this voxel as a neighbor for all its valid free neighbors
                for i, free_neighbor in enumerate(free_neighbor_indices):
                    free_neighbor_tuple = tuple(free_neighbor)
                    if free_neighbor_tuple not in self.dense_graph:
                        self.dense_graph[free_neighbor_tuple] = []
                    if center_index_tuple not in [
                        n for n, _ in self.dense_graph[free_neighbor_tuple]
                    ]:
                        self.dense_graph[free_neighbor_tuple].append(
                            (center_index_tuple, valid_directions_dist[i])
                        )

    def _remove_edges_bulk(self, center_indices):
        """
        remove edges for all voxels that became occupied
        """

        for center_index in center_indices:
            # remove edges from the voxel
            center_index_tuple = tuple(center_index)
            neighbor_indices = [n for n, dist in self.dense_graph[center_index_tuple]]

            for neighbor in neighbor_indices:
                neighbor_tuple = tuple(neighbor)
                if neighbor_tuple in self.dense_graph:
                    self.dense_graph[neighbor_tuple] = [
                        (n, dist)
                        for n, dist in self.dense_graph[neighbor_tuple]
                        if n != center_index_tuple
                    ]

                    if not self.dense_graph[neighbor_tuple]:
                        del self.dense_graph[neighbor_tuple]

            del self.dense_graph[center_index_tuple]
