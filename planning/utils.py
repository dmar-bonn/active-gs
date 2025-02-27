import numpy as np
import heapq
from scipy.special import comb

from utils.operations import *
import time


def select_points_within_cone(
    point, normal, d_close, d_far, cosine_sim, free_points, voxel_map, pitch_angle
):
    positions = []
    views = []

    dist_vectors = point.unsqueeze(0) - free_points
    distances = torch.linalg.norm(dist_vectors, dim=-1)
    distance_mask = (distances <= d_far) & (distances >= d_close)

    view_vectors = dist_vectors / distances.unsqueeze(1)

    if pitch_angle is not None:  # lift vectos to specified pitch
        pitch_angle = torch.tensor(pitch_angle).float()
        cos_pitch = torch.cos(pitch_angle)
        cos_pitch = torch.clamp(cos_pitch, min=1e-8)
        sin_pitch = torch.sin(pitch_angle)
        xy_magnitude = torch.norm(view_vectors[:, :2], p=2, dim=1, keepdim=True)
        z_component = xy_magnitude * sin_pitch / cos_pitch
        view_vectors = torch.cat([view_vectors[:, :2], z_component], dim=1)
        view_vectors = torch.nn.functional.normalize(view_vectors, p=2, dim=1)

    # frontier voxel
    if torch.all(normal == 0):
        normal = voxel_map.check_visible_direction(point)
        if normal is not None:
            normal = normal / normal.norm()
        else:  # no available candidates
            return positions, views
    else:
        normal = normal / normal.norm()

    angle_cosine = torch.sum(view_vectors * -normal, dim=1)
    angle_mask = angle_cosine >= cosine_sim
    mask = distance_mask & angle_mask
    positions = free_points[mask]
    views = view_vectors[mask]

    return positions, views


def cal_flight_time(path_length, flight_speed):
    ## constant velocity model
    return path_length / flight_speed


def inplace_rotation(point, pitch_angle=None, num=1):
    Ts = repeat(np.eye(4), "h w -> n h w", n=num)
    Ts[:, :3, 3] = point
    Ts[:, :3, :3] = random_rotation(num, pitch_angle)
    return torch.tensor(Ts).type(torch.float32)


class PathPlanner:
    def __init__(self):
        pass

    def final_output(self, goal_indices, paths, travel_distances):
        path_list = []
        travel_distance_list = []
        for goal_index in goal_indices:
            if tuple(goal_index) in paths:
                path_list.append(paths[tuple(goal_index)])
                travel_distance_list.append(travel_distances[tuple(goal_index)])
            else:
                path_list.append([])
                travel_distance_list.append(float("inf"))

        return path_list, travel_distance_list

    # a star: find shortest path from start voxel to all goal voxels
    def search_goal(self, start, goals, voxel_map):
        t1 = time.time()
        size = voxel_map.size.cpu().numpy()
        dim = voxel_map.dim.cpu().numpy()
        bbox = voxel_map.bbox.cpu().numpy()
        voxel_centers = voxel_map.voxel_centers.cpu().numpy().reshape((*dim, 3))
        graph = voxel_map.graph.dense_graph

        # Compute distance to the nearest goal
        start_index = tuple(np.floor((start - bbox[0]) / size).astype(int))
        goal_indices = np.array(
            [np.floor((goal - bbox[0]) / size).astype(int) for goal in goals]
        )
        # pdb.set_trace()

        # A* implementation: priority queue with heuristic (F = G + H)
        distances = {node: float("inf") for node in graph}  # G score
        distances[start_index] = 0
        priority_queue = [(0, start_index)]  # F = G + H
        parents = {start_index: None}

        remaining_goals = set(
            [tuple(goal) for goal in goal_indices if tuple(goal) in graph]
        )
        paths = {tuple(goal): [] for goal in remaining_goals}
        travel_distances = {tuple(goal): float("inf") for goal in remaining_goals}

        def heuristic(current_voxel):
            current_voxel_center = voxel_centers[tuple(current_voxel)]
            h = np.min(np.linalg.norm(goals - current_voxel_center, axis=1))
            return h

        while priority_queue and remaining_goals:
            current_f_score, current_node = heapq.heappop(priority_queue)

            # If current node is one of the goals, stop
            if tuple(current_node) in remaining_goals:
                remaining_goals.remove(tuple(current_node))
                # Store the shortest path to this goal
                path = []
                node = current_node
                while node is not None:
                    path.append(node)
                    node = parents.get(node)
                path.reverse()
                paths[tuple(current_node)] = path
                travel_distances[tuple(current_node)] = distances[current_node]

                if not remaining_goals:  # All goals found
                    break

            # Continue exploring neighbors
            for neighbor, weight in graph[current_node]:
                g_score = (
                    distances[current_node] + weight
                )  # G = G(current) + dist(neighbor)

                if g_score < distances[neighbor]:
                    distances[neighbor] = g_score
                    parents[neighbor] = current_node

                    # F = G + H, add to priority queue
                    f_score = g_score + heuristic(neighbor)
                    heapq.heappush(priority_queue, (f_score, neighbor))

        path_list, travel_distance_list = self.final_output(
            goal_indices, paths, travel_distances
        )
        t2 = time.time()
        print(t2 - t1)
        return path_list, travel_distance_list

    # dijkstra algorithm: find shortest path from start voxel to all voxels within range
    def search_range(self, start, plan_range, voxel_map):
        t1 = time.time()
        size = voxel_map.size.cpu().numpy()
        dim = voxel_map.dim.cpu().numpy()
        bbox = voxel_map.bbox.cpu().numpy()
        voxel_centers = voxel_map.voxel_centers.cpu().numpy()
        graph = voxel_map.graph.dense_graph

        range_from_start = np.linalg.norm(voxel_centers - start, axis=1)
        free_mask = voxel_map.free_mask_w_margin.cpu().numpy()
        within_range = range_from_start <= plan_range
        start_index = tuple(np.floor((start - bbox[0]) / size).astype(int))
        valid_mask = free_mask & within_range
        valid_mask = valid_mask.reshape(*dim)

        distances = {node: float("inf") for node in graph}
        distances[start_index] = 0
        priority_queue = [(0, start_index)]
        parents = {start_index: None}

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            if current_distance > distances[current_node]:
                continue

            for neighbor, weight in graph[current_node]:
                # research in range
                if valid_mask[neighbor[0], neighbor[1], neighbor[2]]:
                    distance = current_distance + weight

                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        parents[neighbor] = current_node
                        heapq.heappush(priority_queue, (distance, neighbor))

        t2 = time.time()
        print(t2 - t1)

        indices = np.array(list(distances.keys()))
        dists = np.array(list(distances.values()))

        reachable_mask = dists < 1000
        indices = indices[reachable_mask]
        dists = dists[reachable_mask]
        positions = voxel_map.index_2_xyz(indices).cpu()
        return positions, indices, dists, parents


def rotation_from_z(z_axis):
    y_axis = torch.tensor([0.0, 0.0, -1.0])

    # check if the Z-axis is collinear with the Y-axis (i.e., Z == Y or Z == -Y)
    if torch.allclose(z_axis, y_axis) or torch.allclose(z_axis, -y_axis):
        # special case: Z-axis is already aligned with the Y-axis
        # set X-axis as [1, 0, 0] and Z-axis as [0, 0, -1]
        x_axis = torch.tensor([1.0, 0.0, 0.0])
        # z_axis = torch.tensor([0.0, 0.0, -1.0])  # Flip Z to point upwards
        y_axis_new = torch.cross(z_axis, x_axis)
        # y_axis_new = torch.tensor([0.0, -1.0, 0.0])  # Y-axis remains downward
    else:
        # compute the X-axis using cross product (Y × Z)
        x_axis = torch.cross(y_axis, z_axis)
        x_axis = x_axis / torch.norm(x_axis)  # Normalize X-axis

        # recompute Y-axis to ensure orthogonality (Z × X)
        y_axis_new = torch.cross(z_axis, x_axis)

    y_axis_new = y_axis_new / torch.norm(y_axis_new, dim=-1, keepdim=True)
    # construct the rotation matrix
    rotation_matrix = torch.column_stack((x_axis, y_axis_new, z_axis))
    return rotation_matrix


# assume no roll angles
def rotation_from_z_batch(z_axis_batch):
    z_axis_batch = z_axis_batch / torch.norm(z_axis_batch, dim=-1, keepdim=True)
    # shape of z_axis_batch: (batch_size, 3)
    batch_size = z_axis_batch.shape[0]
    y_axis = torch.tensor([0.0, 0.0, -1.0], device=z_axis_batch.device).expand(
        batch_size, -1
    )

    # check if the Z-axis is collinear with the Y-axis or the negative Y-axis
    is_collinear = torch.all(
        torch.isclose(z_axis_batch, y_axis, atol=1e-3), dim=1
    ) | torch.all(torch.isclose(z_axis_batch, -y_axis, atol=1e-3), dim=1)

    # prepare X and Y axes based on collinearity
    x_axis = torch.where(
        is_collinear.unsqueeze(-1),
        torch.tensor([1.0, 0.0, 0.0], device=z_axis_batch.device).expand(
            batch_size, -1
        ),
        torch.cross(y_axis, z_axis_batch, dim=-1),
    )

    # normalize the X-axis
    x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True)

    # Recompute Y-axis for all cases to ensure orthogonality (Z × X)
    y_axis_new = torch.cross(z_axis_batch, x_axis, dim=-1)
    y_axis_new = y_axis_new / torch.norm(y_axis_new, dim=-1, keepdim=True)

    # construct the rotation matrix
    rotation_matrix = torch.stack((x_axis, y_axis_new, z_axis_batch), dim=-1)
    return rotation_matrix


def bezier_curve(control_points, num_points=100):
    n = len(control_points) - 1
    t = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, len(control_points[0])))

    for i in range(n + 1):
        curve += np.outer(comb(n, i) * (t**i) * ((1 - t) ** (n - i)), control_points[i])

    return curve


def angle_between_vec(v1, v2):
    v1 = v1 / v1.norm(p=2, dim=-1, keepdim=True)
    v2 = v2 / v2.norm(p=2, dim=-1, keepdim=True)

    # compute the dot product between v1 and v2
    dot = torch.sum(v1 * v2, dim=-1)

    # clamp the dot product to avoid numerical errors
    dot = torch.clamp(dot, -1.0, 1.0)

    # compute the angle between the vectors
    theta = torch.acos(dot)
    return theta


def slerp(v1, v2, t):
    """
    perform spherical linear interpolation (SLERP) between two unit vectors v1 and v2 at time t.
    """
    # normalize the input vectors
    theta_0 = angle_between_vec(v1, v2)
    # compute sin(theta_0) and sin(theta)
    sin_theta_0 = torch.sin(theta_0)

    # v1 and v2 are same directions
    if theta_0 < 1e-3:
        return v2.repeat(len(t), 1)

    # compute the interpolated vectors
    sin_t_theta = torch.sin(t.unsqueeze(-1) * theta_0)  # Broadcasting t
    sin_1_minus_t_theta = torch.sin(
        (1 - t.unsqueeze(-1)) * theta_0
    )  # broadcasting (1 - t)

    v_interpolated = (
        sin_1_minus_t_theta * v1 + sin_t_theta * v2
    ) / sin_theta_0.unsqueeze(-1)

    # normalize the interpolated vector
    return v_interpolated / v_interpolated.norm(p=2, dim=-1, keepdim=True)


def wp2path(
    start_rotation,
    goal_rotation,
    waypoints,
    distance_thre=0.05,
    angle_thre=0.1,
):
    start_view_direction = start_rotation[:, 2]  # z direction
    goal_view_direction = goal_rotation[:, 2]
    angle_distance = angle_between_vec(start_view_direction, goal_view_direction)
    num_sample_angle = torch.ceil(angle_distance / angle_thre).long()

    if len(waypoints) == 1:  # inplace rotation
        path_length = torch.tensor(0)
        num_sample = num_sample_angle
        interpolated_positions = torch.tensor(waypoints[-1]).repeat(num_sample, 1)
    else:
        diffs = torch.tensor(waypoints[1:] - waypoints[:-1])
        path_length = torch.sum(torch.norm(diffs, dim=1))
        num_sample_xyz = torch.ceil(path_length / distance_thre).long()
        num_sample = max(num_sample_xyz, num_sample_angle)
        interpolated_positions = bezier_curve(waypoints, num_points=num_sample)

    t = torch.linspace(0, 1, num_sample)
    interpolated_view_directions = slerp(start_view_direction, goal_view_direction, t)
    interpolated_rotations = rotation_from_z_batch(interpolated_view_directions)

    path = torch.eye(4).repeat(num_sample, 1, 1)
    path[:, :3, 3] = torch.tensor(interpolated_positions)
    path[:, :3, :3] = torch.tensor(interpolated_rotations)

    return path, path_length.item()
