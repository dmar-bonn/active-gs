import queue
from scipy.interpolate import CubicSpline
import cv2
import numpy as np
import open3d as o3d
import torch
import torch.multiprocessing as mp
import math
from einops import repeat
import pdb
import matplotlib.pyplot as plt
from scipy.special import comb


cv_gl = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
gl_cv = np.linalg.inv(cv_gl)


class Frustum:
    def __init__(self, line_set, view_dir=None, view_dir_behind=None, size=None):
        self.line_set = line_set
        self.view_dir = view_dir
        self.view_dir_behind = view_dir_behind
        self.size = size

    def update_pose(self, pose):
        points = np.asarray(self.line_set.points)
        points_hmg = np.hstack([points, np.ones((points.shape[0], 1))])
        points = (pose @ points_hmg.transpose())[0:3, :].transpose()

        base = np.array([[0.0, 0.0, 0.0]]) * self.size
        base_hmg = np.hstack([base, np.ones((base.shape[0], 1))])
        cameraeye = pose @ base_hmg.transpose()
        cameraeye = cameraeye[0:3, :].transpose()
        eye = cameraeye[0, :]

        base_behind = np.array([[0.0, -2.5, -30.0]]) * self.size
        base_behind_hmg = np.hstack([base_behind, np.ones((base_behind.shape[0], 1))])
        cameraeye_behind = pose @ base_behind_hmg.transpose()
        cameraeye_behind = cameraeye_behind[0:3, :].transpose()
        eye_behind = cameraeye_behind[0, :]

        center = np.mean(points[1:, :], axis=0)
        up = points[2] - points[4]

        self.view_dir = (center, eye, up, pose)
        self.view_dir_behind = (center, eye_behind, up, pose)

        self.center = center
        self.eye = eye
        self.up = up


def bezier_curve(control_points, num_points=100):
    n = len(control_points) - 1
    t = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, len(control_points[0])))

    for i in range(n + 1):
        curve += np.outer(comb(n, i) * (t**i) * ((1 - t) ** (n - i)), control_points[i])

    return curve


def create_path(waypoints, color=[1, 0, 0]):
    interpolated_path = bezier_curve(waypoints, num_points=50)

    lines = []
    # Connect each point to the next one
    for i in range(len(interpolated_path) - 1):
        lines.append([i, i + 1])

    colors = [color for i in range(len(lines))]
    canonical_line_set = o3d.geometry.LineSet()
    canonical_line_set.points = o3d.utility.Vector3dVector(interpolated_path)
    canonical_line_set.lines = o3d.utility.Vector2iVector(lines)
    canonical_line_set.colors = o3d.utility.Vector3dVector(colors)
    return canonical_line_set


def create_voxel(voxel_centers, voxel_size):
    points = np.array(
        [
            [0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5],
            [0.5, -0.5, 0.5],
            [0.5, -0.5, -0.5],
            [-0.5, 0.5, 0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [-0.5, -0.5, -0.5],
        ]
    )
    points[:, 0] *= voxel_size[0]
    points[:, 1] *= voxel_size[1]
    points[:, 2] *= voxel_size[2]

    voxel_centers = voxel_centers.reshape(-1, 3)  # maske sure shape follow (N, 3)
    vertices = voxel_centers[:, None, :] + points[None, :, :]
    vertices = vertices.reshape(-1, 3)
    num_voxels = voxel_centers.shape[0]

    base_indices = np.arange(num_voxels) * 8
    lines = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [1, 3],
            [1, 5],
            [2, 3],
            [2, 6],
            [3, 7],
            [4, 5],
            [4, 6],
            [5, 7],
            [6, 7],
        ]
    )
    edges = (base_indices[:, None, None] + lines[None, :, :]).reshape(-1, 2)

    z_values = voxel_centers[:, 2]
    z_min = np.min(z_values)
    z_max = np.max(z_values)
    z_normalized = (z_values - z_min) / (z_max - z_min)
    colormap = plt.get_cmap("plasma")
    # color_factor = colormap(voxel_util)[:, :3]
    color_factor = colormap(z_normalized)[:, :3]
    colors = repeat(color_factor, "n c -> n r c", r=12).reshape(-1, 3)
    # colors = np.tile(np.array([[1, 1, 1]]), (edges.shape[0], 1))

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(edges)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def create_frustum(pose, frusutum_color=[0, 1, 0], size=0.02):
    points = (
        np.array(
            [
                [0.0, 0.0, 0],
                [1.0, -0.5, 2],
                [-1.0, -0.5, 2],
                [1.0, 0.5, 2],
                [-1.0, 0.5, 2],
            ]
        )
        * size
    )

    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [2, 4], [3, 4]]
    colors = [frusutum_color for i in range(len(lines))]

    canonical_line_set = o3d.geometry.LineSet()
    canonical_line_set.points = o3d.utility.Vector3dVector(points)
    canonical_line_set.lines = o3d.utility.Vector2iVector(lines)
    canonical_line_set.colors = o3d.utility.Vector3dVector(colors)
    frustum = Frustum(canonical_line_set, size=size)
    frustum.update_pose(pose)
    return frustum


def get_latest_queue(q):
    message = None
    while True:
        try:
            message_latest = q.get_nowait()
            if message is not None:
                del message
            message = message_latest
        except queue.Empty:
            if q.qsize() < 1:
                break
    return message


class Packet_vis2main:
    flag_pause = None


def getWorld2View2(R, t, translate=torch.tensor([0.0, 0.0, 0.0]), scale=1.0):
    translate = translate.to(R.device)
    Rt = torch.zeros((4, 4), device=R.device)
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = torch.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.linalg.inv(C2W)
    return Rt


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = -(zfar + znear) / (zfar - znear)
    P[2, 3] = -2 * (zfar * znear) / (zfar - znear)
    return P


def getProjectionMatrix2(znear, zfar, cx, cy, fx, fy, W, H):
    left = ((2 * cx - W) / W - 1.0) * W / 2.0
    right = ((2 * cx - W) / W + 1.0) * W / 2.0
    top = ((2 * cy - H) / H + 1.0) * H / 2.0
    bottom = ((2 * cy - H) / H - 1.0) * H / 2.0
    left = znear / fx * left
    right = znear / fx * right
    top = znear / fy * top
    bottom = znear / fy * bottom
    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def c2w_to_lookat(c2w):
    eye = c2w[:3, 3]  # The last column represents translation (position)

    # Extract the forward vector (negative Z-axis of the model matrix)
    forward_vector = -c2w[:3, 2]  # Assuming column-major order

    # Calculate the look-at point (center)
    center = eye + forward_vector  # Point in front of the camera

    # Extract the up vector (Y-axis of the model matrix)
    up = -c2w[:3, 1]  # Assuming column-major order
    return [eye, center, up, c2w]


def model_matrix_to_extrinsic_matrix(model_matrix):
    return np.linalg.inv(model_matrix @ gl_cv)


def create_camera_intrinsic_from_size(width=1024, height=768, hfov=60.0, vfov=60.0):
    fx = (width / 2.0) / np.tan(np.radians(hfov) / 2)
    fy = (height / 2.0) / np.tan(np.radians(vfov) / 2)
    fx = fy  # not sure why, but it looks like fx should be governed/limited by fy
    return np.array([[fx, 0, width / 2.0], [0, fy, height / 2.0], [0, 0, 1]])
