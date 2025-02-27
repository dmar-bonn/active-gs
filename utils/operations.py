from jaxtyping import Float, Int64, Bool
import torch
from torch import Tensor
import numpy as np
import cv2
import trimesh
import torchvision.transforms as tf
from scipy.spatial import cKDTree as KDTree
from scipy.spatial.transform import Rotation as R
from einops import rearrange, repeat
import copy
import torchvision.transforms as tf
from math import isqrt, tan
from einops import einsum, rearrange, reduce, repeat
from torch.functional import norm
import torch.nn.functional as F
import math
import open3d as o3d
import pdb

# from diff_gaussian_rasterization import (
#     GaussianRasterizationSettings,
#     GaussianRasterizer,
# )

from diff_gaussian_rasterization_2d import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


opencv_rotation = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])


def open3dmesh_2_trimesh(mesh):
    vertices = mesh.vertices
    faces = mesh.triangles

    vertices_np = np.asarray(vertices)
    faces_np = np.asarray(faces)

    # Create a trimesh.Trimesh object
    mesh_trimesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np)
    return mesh_trimesh


def voxel_filter(dense_pcd):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(dense_pcd)
    # Apply voxel grid downsampling with voxel size of 1cm (0.01 units)
    voxel_size = 0.001  # 0.1cm in meters
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)

    # Convert the downsampled point cloud back to numpy array
    downsampled_pcd = np.asarray(downsampled_pcd.points)
    return downsampled_pcd


def calc_3d_mesh_metric_new(mesh_rec, mesh_gt):
    """
    3D reconstruction metric.

    """

    rec_pc = trimesh.sample.sample_surface(mesh_rec, 1000000)
    rec_pc = np.array(rec_pc[0])
    rec_pc_ds = voxel_filter(rec_pc)
    # rec_pc_tri = trimesh.PointCloud(vertices=rec_pc_ds)

    gt_pc = trimesh.sample.sample_surface(mesh_gt, 1000000)
    gt_pc = np.array(gt_pc[0])
    gt_pc_ds = voxel_filter(gt_pc)
    # gt_pc_tri = trimesh.PointCloud(vertices=gt_pc_ds)
    accuracy_rec = accuracy(gt_pc_ds, rec_pc_ds)
    completion_rec = completion(gt_pc_ds, rec_pc_ds)
    completion_ratio_rec = completion_ratio(gt_pc_ds, rec_pc_ds)
    accuracy_rec *= 100  # convert to cm
    completion_rec *= 100  # convert to cm
    completion_ratio_rec *= 100  # convert to %

    return accuracy_rec, completion_rec, completion_ratio_rec


def calc_3d_mesh_metric(mesh_rec, mesh_gt, align=False, dist_thres=0.05):
    """
    3D reconstruction metric.

    """

    rec_pc = trimesh.sample.sample_surface(mesh_rec, 500000)
    rec_pc_tri = trimesh.PointCloud(vertices=rec_pc[0])

    gt_pc = trimesh.sample.sample_surface(mesh_gt, 500000)
    gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])

    accuracy_rec = accuracy(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_rec = completion(gt_pc_tri.vertices, rec_pc_tri.vertices)
    chamfer_dist = (accuracy_rec + completion_rec) / 2 
    completion_ratio_rec = completion_ratio(
        gt_pc_tri.vertices, rec_pc_tri.vertices, dist_th=dist_thres
    )
    accuracy_rec *= 100  # convert to cm
    completion_rec *= 100  # convert to cm
    completion_ratio_rec *= 100  # convert to %

    return accuracy_rec, completion_rec, completion_ratio_rec, chamfer_dist


def completion_ratio(gt_points, rec_points, dist_th=0.01):
    gen_points_kd_tree = KDTree(rec_points)
    distances, _ = gen_points_kd_tree.query(gt_points)
    comp_ratio = np.mean((distances < dist_th).astype(np.float32))
    return comp_ratio


def accuracy(gt_points, rec_points):
    gt_points_kd_tree = KDTree(gt_points)
    distances, _ = gt_points_kd_tree.query(rec_points)
    acc = np.mean(distances)
    return acc


def completion(gt_points, rec_points):
    gt_points_kd_tree = KDTree(rec_points)
    distances, _ = gt_points_kd_tree.query(gt_points)
    comp = np.mean(distances)
    return comp


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def normal2curv(normal, mask):
    # normal = normal.detach()
    n = normal.permute([0, 2, 3, 1])
    m = mask.permute([0, 2, 3, 1])
    n = torch.nn.functional.pad(n[None], [0, 0, 1, 1, 1, 1], mode="replicate")
    m = torch.nn.functional.pad(
        m[None].to(torch.float32), [0, 0, 1, 1, 1, 1], mode="replicate"
    ).to(torch.bool)
    n_c = (n[:, 1:-1, 1:-1, :]) * m[:, 1:-1, 1:-1, :]
    n_u = (n[:, :-2, 1:-1, :] - n_c) * m[:, :-2, 1:-1, :]
    n_l = (n[:, 1:-1, :-2, :] - n_c) * m[:, 1:-1, :-2, :]
    n_b = (n[:, 2:, 1:-1, :] - n_c) * m[:, 2:, 1:-1, :]
    n_r = (n[:, 1:-1, 2:, :] - n_c) * m[:, 1:-1, 2:, :]
    curv = (n_u + n_l + n_b + n_r)[0]
    curv = curv.permute([2, 0, 1]) * mask
    curv = curv.norm(1, 0, True)
    return curv


# def random_rotation(n, roll_pitch_yaw, opencv=True):
#     roll, pitch, yaw = roll_pitch_yaw
#     if roll is None:
#         random_z_rot = np.random.uniform(0, 2 * np.pi, n)
#     else:
#         random_z_rot = np.ones(n) * roll

#     if pitch is None:
#         random_x_rot = np.random.uniform(-0.5 * np.pi, 0.5 * np.pi, n)
#     else:
#         random_x_rot = np.ones(n) * pitch


#     if yaw is None:
#         random_y_rot = np.random.uniform(0, 2 * np.pi, n)
#     else:
#         random_y_rot = np.one(n) * yaw
#     eulers = np.stack((random_z_rot, random_x_rot, random_y_rot), axis=-1)
#     rotation_matrix = R.from_euler("zxy", eulers).as_matrix()
#     if opencv:
#         rotation_matrix = opencv_rotation @ rotation_matrix
#     return rotation_matrix


# def decode_rotation_from_z(z_axis_batch, roll_pitch_yaw, opencv=True):
#     roll, pitch, yaw = roll_pitch_yaw
#     n = len(z_axis_batch)
#     z_axis_batch = z_axis_batch.numpy()
#     if roll is None:
#         z_rot = np.random.uniform(0, 2 * np.pi, n)
#     else:
#         z_rot = np.ones(n) * roll

#     x_rot = np.arcsin(np.clip(z_axis_batch[:, 2], -1.0, 1.0))
#     y_rot = np.arctan2(z_axis_batch[:, 1], z_axis_batch[:, 0])
#     eulers = np.stack((z_rot, x_rot, y_rot), axis=-1)
#     # pdb.set_trace()
#     rotation_matrix = R.from_euler("zxy", eulers).as_matrix()

#     if opencv:
#         rotation_matrix = opencv_rotation @ rotation_matrix

#     return torch.tensor(rotation_matrix)


def random_rotation(n, pitch_angle, opencv=True):
    points = np.random.randn(n, 3)  # Sample from normal distribution
    points = points / np.clip(np.linalg.norm(points, axis=1, keepdims=True), 1e-8, None)

    z_rot = np.zeros(n)

    if pitch_angle is None:
        x_rot = np.arcsin(points[:, 2])
    else:
        x_rot = np.ones(n) * pitch_angle

    y_rot = np.arctan2(points[:, 1], points[:, 0])

    eulers = np.stack((z_rot, x_rot, y_rot), axis=-1)
    rotation_matrix = R.from_euler("zxy", eulers).as_matrix()
    if opencv:
        rotation_matrix = opencv_rotation @ rotation_matrix
    return rotation_matrix


# def random_point_in_sphere(radius, n):
#     # Generate a random radius r
#     u1 = np.random.uniform(0, 1, n)
#     r = radius * np.cbrt(u1)

#     # Generate random angles theta and phi
#     u2 = np.random.uniform(0, 1, n)
#     theta = 2 * np.pi * u2

#     u3 = np.random.uniform(0, 1, n)
#     phi = np.arccos(2 * u3 - 1)

#     # Convert spherical coordinates to Cartesian coordinates
#     x = r * np.sin(phi) * np.cos(theta)
#     y = r * np.sin(phi) * np.sin(theta)
#     z = r * np.cos(phi)

#     return np.stack((x, y, z), axis=-1)


def convert_poses(
    poses: Float[Tensor, "batch 18"],
) -> tuple[
    Float[Tensor, "batch 4 4"],  # extrinsics
    Float[Tensor, "batch 3 3"],  # intrinsics
]:
    """convert replica formate pose to 4x4 matrix"""

    b, _ = poses.shape

    # Convert the intrinsics to a 3x3 normalized K matrix.
    intrinsics = torch.eye(3, dtype=torch.float32)
    intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
    fx, fy, cx, cy = poses[:, :4].T
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy

    # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
    w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
    w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
    # return w2c.inverse(), intrinsics
    return w2c, intrinsics


def clone_obj(obj):
    clone_obj = copy.deepcopy(obj)
    for attr in clone_obj.__dict__.keys():
        # check if its a property
        if hasattr(clone_obj.__class__, attr) and isinstance(
            getattr(clone_obj.__class__, attr), property
        ):
            continue
        if isinstance(getattr(clone_obj, attr), torch.Tensor):
            setattr(clone_obj, attr, getattr(clone_obj, attr).detach().clone())
    return clone_obj


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


# def depth2normal(depth_map):
#     """generate normal map from depth image"""
#     _, h, w = depth_map.shape
#     depth_map = depth_map.unsqueeze(1)
#     device = depth_map.device
#     delzdelxkernel = torch.tensor(
#         [
#             [0.00000, 0.00000, 0.00000],
#             [-1.00000, 0.00000, 1.00000],
#             [0.00000, 0.00000, 0.00000],
#         ]
#     )
#     delzdelxkernel = delzdelxkernel.view(1, 1, 3, 3).to(device)
#     delzdelx = F.conv2d(depth_map, delzdelxkernel, padding="same")

#     delzdelykernel = torch.tensor(
#         [
#             [0.00000, -1.00000, 0.00000],
#             [0.00000, 0.00000, 0.00000],
#             [0.0000, 1.00000, 0.00000],
#         ]
#     )
#     delzdelykernel = delzdelykernel.view(1, 1, 3, 3).to(device)

#     delzdely = F.conv2d(depth_map, delzdelykernel, padding="same")

#     delzdelz = torch.ones(delzdely.shape, dtype=torch.float64).to(device)
#     # print('kernel',delzdelx.shape)
#     surface_norm = torch.stack((-2 * delzdelx, -2 * delzdely, delzdelz), 2)
#     surface_norm = torch.div(surface_norm, norm(surface_norm, dim=2)[:, :, None, :, :])
#     return surface_norm.view(-1, h, w)


def get_smooth_depth(depth, tolerance=0.5):
    invalid_mask = depth < 0.0
    valid_depth_image = np.copy(depth)
    valid_depth_image[invalid_mask] = np.nan
    filtered_depth = cv2.bilateralFilter(
        np.nan_to_num(valid_depth_image), 15, tolerance, 20
    )
    filtered_depth[invalid_mask] = -1.0
    return filtered_depth


def depth2normal(depth, mask, fov):
    camD = depth.permute([1, 2, 0])
    mask = mask.permute([1, 2, 0])
    shape = camD.shape
    device = camD.device
    h, w, _ = torch.meshgrid(
        torch.arange(0, shape[0], device=device, dtype=torch.float32),
        torch.arange(0, shape[1], device=device, dtype=torch.float32),
        torch.arange(0, shape[2], device=device, dtype=torch.float32),
        indexing="ij",
    )
    p = torch.cat([w, h], axis=-1)

    p[..., 0:1] -= 0.5 * shape[1]
    p[..., 1:2] -= 0.5 * shape[0]
    p *= camD
    K00 = fov2focal(fov[0], shape[0])
    K11 = fov2focal(fov[1], shape[1])
    K = torch.tensor([K00, 0, 0, K11], device=device).reshape([2, 2])
    Kinv = torch.inverse(K)
    p = p @ Kinv.t()
    camPos = torch.cat([p, camD], -1)

    p_padded = torch.nn.functional.pad(
        camPos[None], [0, 0, 1, 1, 1, 1], mode="replicate"
    )
    mask_padded = torch.nn.functional.pad(
        mask[None].to(torch.float32), [0, 0, 1, 1, 1, 1], mode="replicate"
    ).to(torch.bool)

    p_c = p_padded[:, 1:-1, 1:-1, :] * mask_padded[:, 1:-1, 1:-1, :]
    p_u = (p_padded[:, :-2, 1:-1, :] - p_c) * mask_padded[:, :-2, 1:-1, :]
    p_l = (p_padded[:, 1:-1, :-2, :] - p_c) * mask_padded[:, 1:-1, :-2, :]
    p_b = (p_padded[:, 2:, 1:-1, :] - p_c) * mask_padded[:, 2:, 1:-1, :]
    p_r = (p_padded[:, 1:-1, 2:, :] - p_c) * mask_padded[:, 1:-1, 2:, :]

    n_ul = torch.cross(p_u, p_l)
    n_ur = torch.cross(p_r, p_u)
    n_br = torch.cross(p_b, p_r)
    n_bl = torch.cross(p_l, p_b)

    n = n_ul + n_ur + n_br + n_bl
    n = n[0]

    n = torch.nn.functional.normalize(n, dim=-1)

    n = (n * mask).permute([2, 0, 1])
    return n


def cal_scale_factor(intrinsic, resolution):
    H, W = resolution
    fov = get_fov(intrinsic.unsqueeze(0)).squeeze(0)
    fovx, fovy = fov
    x_unit = fovx / W
    y_unit = fovy / H
    x_scale = 2 * tan(x_unit / 2)
    y_scale = 2 * tan(y_unit / 2)
    return torch.min(torch.tensor([x_scale, y_scale]))


def rescale_and_crop(rgb, depth, intrinsic, resolution):
    """rescale and crop rgb-d image and its intrinsic"""

    _, h_in, w_in = rgb.shape
    _, h_in_d, w_in_d = depth.shape
    assert (h_in == h_in_d) and (w_in == w_in_d)

    h_out, w_out = resolution
    scale_factor = max(h_out / h_in, w_out / w_in)

    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    assert h_scaled == h_out or w_scaled == w_out

    rgb_resize = tf.functional.resize(rgb, (h_scaled, w_scaled))
    depth_resize = tf.functional.resize(depth, (h_scaled, w_scaled))

    row = (h_scaled - h_out) // 2
    col = (w_scaled - w_out) // 2
    rgb_out = rgb_resize[:, row : row + h_out, col : col + w_out]
    depth_out = depth_resize[:, row : row + h_out, col : col + w_out]

    intrinsic_out = intrinsic.clone()
    intrinsic_out[0, 0] *= w_scaled / w_out  # fx
    intrinsic_out[1, 1] *= h_scaled / h_out  # fy
    return rgb_out, depth_out, intrinsic_out


def quaternion_to_matrix(q):
    r, x, y, z = q.split(1, -1)
    # R = torch.eye(4).expand([len(q), 4, 4]).to(q.device)
    R = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - r * z),
            2 * (x * z + r * y),
            2 * (x * y + r * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - r * x),
            2 * (x * z - r * y),
            2 * (y * z + r * x),
            1 - 2 * (x * x + y * y),
        ],
        -1,
    ).reshape([len(q), 3, 3])
    return R


# def quaternion_to_matrix(
#     quaternions: Float[Tensor, "*batch 4"],
#     eps: float = 1e-8,
# ) -> Float[Tensor, "*batch 3 3"]:
#     # Order changed to match scipy format!
#     r, i, j, k = torch.unbind(quaternions, dim=-1)
#     two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

#     o = torch.stack(
#         (
#             1 - two_s * (j * j + k * k),
#             two_s * (i * j - k * r),
#             two_s * (i * k + j * r),
#             two_s * (i * j + k * r),
#             1 - two_s * (i * i + k * k),
#             two_s * (j * k - i * r),
#             two_s * (i * k - j * r),
#             two_s * (j * k + i * r),
#             1 - two_s * (i * i + j * j),
#         ),
#         -1,
#     )
#     return rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def _standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """

    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """

    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """

    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return _standardize_quaternion(out)


def build_covariance(
    scale: Float[Tensor, "*#batch 3"],
    rotation_xyzw: Float[Tensor, "*#batch 4"],
) -> Float[Tensor, "*batch 3 3"]:
    """calculate covariance matrix from scale and quaternion"""

    scale = scale.diag_embed()
    rotation = quaternion_to_matrix(rotation_xyzw)
    return (
        rotation
        @ scale
        @ rearrange(scale, "... i j -> ... j i")
        @ rearrange(rotation, "... i j -> ... j i")
    )


def sample_image_grid(
    shape: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
) -> tuple[
    Float[Tensor, "*shape dim"],  # float coordinates (xy indexing)
    Int64[Tensor, "*shape dim"],  # integer indices (ij indexing)
]:
    """Get normalized (range 0 to 1) coordinates and integer indices for an image."""

    # Each entry is a pixel-wise integer coordinate. In the 2D case, each entry is a
    # (row, col) coordinate.
    indices = [torch.arange(length, device=device) for length in shape]
    stacked_indices = torch.stack(torch.meshgrid(*indices, indexing="ij"), dim=-1)

    # Each entry is a floating-point coordinate in the range (0, 1). In the 2D case,
    # each entry is an (x, y) coordinate.
    coordinates = [(idx + 0.5) / length for idx, length in zip(indices, shape)]
    coordinates = reversed(coordinates)
    coordinates = torch.stack(torch.meshgrid(*coordinates, indexing="xy"), dim=-1)

    return coordinates, stacked_indices


def homogenize_points(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Convert batched points (xyz) to (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def homogenize_vectors(
    vectors: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Convert batched vectors (xyz) to (xyz0)."""
    return torch.cat([vectors, torch.zeros_like(vectors[..., :1])], dim=-1)


def transform_rigid(
    homogeneous_coordinates: Float[Tensor, "*#batch dim"],
    transformation: Float[Tensor, "*#batch dim dim"],
) -> Float[Tensor, "*batch dim"]:
    """Apply a rigid-body transformation to points or vectors."""
    return einsum(transformation, homogeneous_coordinates, "... i j, ... j -> ... i")


def transform_cam2world(
    homogeneous_coordinates: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim dim"],
) -> Float[Tensor, "*batch dim"]:
    """Transform points from 3D camera coordinates to 3D world coordinates."""
    return transform_rigid(homogeneous_coordinates, extrinsics)


def transform_world2cam(
    homogeneous_coordinates: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim dim"],
) -> Float[Tensor, "*batch dim"]:
    """Transform points from 3D world coordinates to 3D camera coordinates."""
    return transform_rigid(homogeneous_coordinates, extrinsics.inverse())


def project_camera_space(
    points: Float[Tensor, "*#batch dim"],
    intrinsics: Float[Tensor, "*#batch dim dim"],
    epsilon: float = torch.finfo(torch.float32).eps,
    infinity: float = 1e8,
) -> Float[Tensor, "*batch dim-1"]:
    """project points in camera coordinate to image plane"""

    points = points / (points[..., -1:] + epsilon)
    points = points.nan_to_num(posinf=infinity, neginf=-infinity)
    points = einsum(intrinsics, points, "... i j, ... j -> ... i")
    return points[..., :-1]


def project(
    points: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim+1 dim+1"],
    intrinsics: Float[Tensor, "*#batch dim dim"],
    epsilon: float = torch.finfo(torch.float32).eps,
) -> tuple[
    Float[Tensor, "*batch dim-1"],  # xy coordinates
    Bool[Tensor, " *batch"],  # whether points are in front of the camera
]:
    """project point in world coordinate to image plane"""

    points = homogenize_points(points)
    points = transform_world2cam(points, extrinsics)[..., :-1]
    in_front_of_camera = points[..., -1] >= 0
    return project_camera_space(points, intrinsics, epsilon=epsilon), in_front_of_camera


def unproject(
    coordinates: Float[Tensor, "*#batch dim"],
    z: Float[Tensor, "*#batch"],
    intrinsics: Float[Tensor, "*#batch dim+1 dim+1"],
) -> Float[Tensor, "*batch dim+1"]:
    """Unproject 2D camera coordinates with the given Z values."""

    # Apply the inverse intrinsics to the coordinates.
    coordinates = homogenize_points(coordinates)
    ray_directions = einsum(
        intrinsics.inverse(), coordinates, "... i j, ... j -> ... i"
    )

    # Apply the supplied depth values.
    return ray_directions * z[..., None]


# def normal2rotation(normals):
#     u_batch = normals
#     u_norm = np.linalg.norm(u_batch, axis=1, keepdims=True)
#     u_batch = u_batch / u_norm

#     # Choose an arbitrary vector that is not parallel to u_batch
#     v = np.array([0.0, 0.0, 1.0])
#     v_batch = np.tile(v, (u_batch.shape[0], 1))

#     # Check if any u_batch vectors are parallel to (0, 0, 1) and handle them
#     parallel_mask = np.allclose(u_batch, v_batch, atol=1e-6)
#     v_batch[parallel_mask] = np.array([0.0, 1.0, 0.0])

#     # Compute the cross product to get perpendicular vectors
#     v2_batch = np.cross(u_batch, v_batch)

#     # Normalize the perpendicular vectors
#     v2_norm = np.linalg.norm(v2_batch, axis=1, keepdims=True)
#     v2_batch = v2_batch / v2_norm

#     # Compute the cross product to get the third vectors
#     v3_batch = np.cross(u_batch, v2_batch)

#     # Normalize the third vectors
#     v3_norm = np.linalg.norm(v3_batch, axis=1, keepdims=True)
#     v3_batch = v3_batch / v3_norm

#     stacked = np.column_stack((u_batch, v2_batch, v3_batch))


#     # Reshape to get the desired (N, 3, 3) shape
#     return stacked.reshape(u_batch.shape[0], u_batch.shape[1], -1)


def normal2rotation(z):
    batch_size = z.shape[0]
    z = z / z.norm(dim=1, keepdim=True)

    # Generate a reference vector
    ref_vector = torch.tensor([1.0, 0.0, 0.0], device=z.device).repeat(batch_size, 1)
    parallel_mask = (torch.abs(z[:, 0]) > 0.99).unsqueeze(1)
    ref_vector[parallel_mask[:, 0]] = torch.tensor([0.0, 1.0, 0.0], device=z.device)

    # Project ref vector onto the plane orthogonal to z and normalize to get x-axis
    projections = (ref_vector * z).sum(dim=1, keepdim=True) * z
    x = ref_vector - projections
    x = x / x.norm(dim=1, keepdim=True)

    # Compute the y-axis as the cross product of z and x
    y = torch.cross(z, x, dim=1)
    y = y / y.norm(dim=1, keepdim=True)
    rotation = torch.stack([x, y, z], dim=-1)
    q = rotmat2quaternion(rotation)
    return q, rotation


# def normal2rotation(n):
#     # construct a random rotation matrix from normal
#     # it would better be positive definite and orthogonal
#     n = torch.nn.functional.normalize(n)
#     # w0 = torch.rand_like(n)
#     w0 = torch.tensor([[1, 0, 0]]).expand(n.shape)
#     R0 = w0 - torch.sum(w0 * n, -1, True) * n
#     R0 *= torch.sign(R0[:, :1])
#     R0 = torch.nn.functional.normalize(R0)
#     R1 = torch.cross(n, R0)

#     # i = 7859
#     # print(R1[i])
#     R1 *= torch.sign(R1[:, 1:2]) * torch.sign(n[:, 2:])
#     # print(R1[i])
#     R = torch.stack([R0, R1, n], -1)
#     # print(R[i], torch.det(R).sum(), torch.trace(R[i]))
#     q = rotmat2quaternion(R)
#     # print(q[i], torch.norm(q[i]))
#     # R = quaternion2rotmat(q)
#     # print(R[i])
#     # for i in range(len(q)):
#     #     if torch.isnan(q[i].sum()):
#     #         print(i)
#     # exit()
#     return q


def quaternion2rotmat(q):
    r, x, y, z = q.split(1, -1)
    # R = torch.eye(4).expand([len(q), 4, 4]).to(q.device)
    R = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - r * z),
            2 * (x * z + r * y),
            2 * (x * y + r * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - r * x),
            2 * (x * z - r * y),
            2 * (y * z + r * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )

    # Reshape to [batch_size, 3, 3]
    R = R.reshape(q.size(0), 3, 3)
    return R


def rotmat2quaternion(R, normalize=True):
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] + 1e-6
    r = torch.sqrt(1 + tr) / 2
    # print(torch.sum(torch.isnan(r)))
    q = torch.stack(
        [
            r,
            (R[:, 2, 1] - R[:, 1, 2]) / (4 * r),
            (R[:, 0, 2] - R[:, 2, 0]) / (4 * r),
            (R[:, 1, 0] - R[:, 0, 1]) / (4 * r),
        ],
        -1,
    )
    if normalize:
        q = torch.nn.functional.normalize(q, dim=-1)
    return q


def get_world_rays(
    coordinates: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim+2 dim+2"],
    intrinsics: Float[Tensor, "*#batch dim+1 dim+1"],
) -> tuple[
    Float[Tensor, "*batch dim+1"],  # origins
    Float[Tensor, "*batch dim+1"],  # directions
]:
    """get rays in world coordinate"""

    # Get camera-space ray directions.
    directions = unproject(
        coordinates,
        torch.ones_like(coordinates[..., 0]),
        intrinsics,
    )
    # directions = directions / directions.norm(dim=-1, keepdim=True)

    # Transform ray directions to world coordinates.
    directions = homogenize_vectors(directions)
    directions = transform_cam2world(directions, extrinsics)[..., :-1]

    # Tile the ray origins to have the same shape as the ray directions.
    origins = extrinsics[..., :-1, -1].broadcast_to(directions.shape)

    return origins, directions


def get_projection_matrix(
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    fov_x: Float[Tensor, " batch"],
    fov_y: Float[Tensor, " batch"],
) -> Float[Tensor, "batch 4 4"]:
    """Maps points in the viewing frustum to (-1, 1) on the X/Y axes and (0, 1) on the Z
    axis. Differs from the OpenGL version in that Z doesn't have range (-1, 1) after
    transformation and that Z is flipped.
    """

    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right

    (b,) = near.shape
    result = torch.zeros((b, 4, 4), dtype=torch.float32, device=near.device)
    result[:, 0, 0] = 2 * near / (right - left)
    result[:, 1, 1] = 2 * near / (top - bottom)
    result[:, 0, 2] = (right + left) / (right - left)
    result[:, 1, 2] = (top + bottom) / (top - bottom)
    result[:, 3, 2] = 1
    result[:, 2, 2] = far / (far - near)
    result[:, 2, 3] = -(far * near) / (far - near)
    return result


def voxel_downsample(point_cloud, voxel_size=0.02, num_points_per_voxel=1):
    # Step 1: Compute voxel indices for each point
    voxel_indices = torch.floor(point_cloud / voxel_size).long()

    # Step 2: Compute unique voxel indices and their corresponding counts
    unique_voxel_indices, inverse_indices = torch.unique(
        voxel_indices, return_inverse=True, dim=0
    )

    # Step 3: Shuffle the inverse indices
    rand_indices = torch.randperm(inverse_indices.size(0), device=point_cloud.device)
    shuffled_inverse_indices = inverse_indices[rand_indices]

    # Step 4: Select one index per unique voxel
    selected_indices = torch.zeros(
        unique_voxel_indices.shape[0], dtype=torch.long, device=point_cloud.device
    )
    selected_indices[shuffled_inverse_indices] = rand_indices

    # Step 5: Ensure only unique indices are selected
    selected_indices = selected_indices.unique()

    return selected_indices


# def voxel_downsample(point_cloud, voxel_size=0.01, num_points_per_voxel=1):
#     # Step 1: Compute voxel indices for each point
#     voxel_indices = torch.floor(point_cloud / voxel_size).int()

#     # Step 2: Compute unique voxel indices and their corresponding counts
#     unique_voxel_indices, inverse_indices = torch.unique(
#         voxel_indices, return_inverse=True, dim=0
#     )

#     rand_indices = torch.randperm(inverse_indices.size(0))
#     shuffled_inverse_indices = inverse_indices[rand_indices]

#     selected_indices = torch.zeros(unique_voxel_indices.shape[0], dtype=torch.long)
#     selected_indices[shuffled_inverse_indices] = rand_indices

#     # Ensure only unique indices are selected
#     selected_indices = selected_indices.unique()
#     return selected_indices

# # Step 4: Create a mask to filter out points from overpopulated voxels
# # Randomly permute points within each voxel group
# perm = torch.randperm(inverse_indices.size(0))
# inverse_indices = inverse_indices[perm]

# # Gather the top num_points_per_voxel points per voxel
# cumulative_counts = torch.cumsum(voxel_counts, 0)
# select_indices = (
#     cumulative_counts.unsqueeze(1) - torch.arange(num_points_per_voxel)
# ).flatten()
# select_mask = (select_indices >= 0) & (select_indices < inverse_indices.size(0))

# # Get valid indices
# valid_indices = perm[select_indices[select_mask]].long()

# Step 5: Return the downsampled point cloud
# downsampled_points = point_cloud[valid_indices]

# return valid_indices


def get_fov(intrinsics: Float[Tensor, "batch 3 3"]) -> Float[Tensor, "batch 2"]:
    intrinsics_inv = intrinsics.inverse()

    def process_vector(vector):
        vector = torch.tensor(vector, dtype=torch.float32, device=intrinsics.device)
        vector = einsum(intrinsics_inv, vector, "b i j, j -> b i")
        return vector / vector.norm(dim=-1, keepdim=True)

    left = process_vector([0, 0.5, 1])
    right = process_vector([1, 0.5, 1])
    top = process_vector([0.5, 0, 1])
    bottom = process_vector([0.5, 1, 1])
    fov_x = (left * right).sum(dim=-1).acos()
    fov_y = (top * bottom).sum(dim=-1).acos()
    return torch.stack((fov_x, fov_y), dim=-1)


# def check_in_frustum(posiotions, extrinsic, intrinsic, near_far):
#     near, far = near_far
#     fov_x, fov_y = get_fov(intrinsic).unbind(dim=-1)
#     projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
#     projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
#     view_matrix = rearrange(extrinsic.inverse(), "b i j -> b j i")
#     return in_frustum(posiotions, projection_matrix[0], view_matrix[0])


# import time


# def render_cuda(
#     extrinsics: Float[Tensor, "batch 4 4"],
#     intrinsics: Float[Tensor, "batch 3 3"],
#     near: Float[Tensor, " batch"],
#     far: Float[Tensor, " batch"],
#     image_shape: tuple[int, int],
#     background_color: Float[Tensor, "3"],
#     gaussian_means: Float[Tensor, "gaussian 3"],
#     gaussian_sh_coefficients: Float[Tensor, "gaussian 3 d_sh"],
#     gaussian_opacities: Float[Tensor, "gaussian"],
#     gaussian_confidences: Float[Tensor, "gaussian"],
#     gaussian_scales,
#     gaussian_rotations,
#     gaussian_normals,
#     scale_invariant: bool = False,
#     front_only=False,
#     require_importance: bool = False,
#     use_sh: bool = False,
#     render_masks=None,
#     weight_thres=0.1,
# ) -> Float[Tensor, "batch 3 height width"]:
#     """main interface to cuda forward rendering kernel"""
#     assert use_sh or gaussian_sh_coefficients.shape[-1] == 1
#     t1 = time.time()
#     front_config = 0.0
#     if front_only:
#         front_config = 1.0

#     importance_config = 0.0
#     if require_importance:
#         importance_config = 1.0

#     # Make sure everything is in a range where numerical issues don't appear.
#     device = gaussian_means.device
#     scale = torch.ones_like(near)
#     if scale_invariant:
#         scale = scale / near
#         near = near * scale
#         far = far * scale
#         scale = scale.unsqueeze(-1)
#         extrinsics = extrinsics.clone()
#         extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale
#         gaussian_scales = gaussian_scales * scale
#         gaussian_means = gaussian_means * scale

#     num_gaussians, _, n = gaussian_sh_coefficients.shape
#     degree = isqrt(n) - 1
#     shs = rearrange(gaussian_sh_coefficients, "g xyz n -> g n xyz").contiguous()

#     b, _, _ = extrinsics.shape
#     h, w = image_shape

#     fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
#     tan_fov_x = (0.5 * fov_x).tan()
#     tan_fov_y = (0.5 * fov_y).tan()

#     xy_ray, _ = sample_image_grid((h, w))
#     xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy").to(intrinsics.device)
#     directions = unproject(
#         xy_ray,
#         torch.ones_like(xy_ray[..., 0]),
#         intrinsics,
#     ).view(h, w, -1, 3)
#     raydir_maps = torch.nn.functional.normalize(directions, dim=-1).to(device)
#     raydir_maps = raydir_maps.permute(2, 3, 0, 1)  # B 3 H W
#     # print(raydir_maps.shape)

#     projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
#     projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
#     view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
#     full_projection = view_matrix @ projection_matrix
#     t2 = time.time()

#     all_rgbs = torch.empty(b, 3, h, w, device=device)
#     all_depths = torch.empty(b, 1, h, w, device=device)
#     all_d2ns = torch.empty(b, 3, h, w, device=device)
#     all_normals = torch.empty(b, 3, h, w, device=device)
#     all_opacities = torch.empty(b, 1, h, w, device=device)
#     all_confidences = torch.empty(b, 1, h, w, device=device)
#     all_importances = torch.zeros(b, num_gaussians, device=device)
#     all_counts = torch.zeros(b, num_gaussians, device=device, dtype=torch.int32)

#     for i in range(b):
#         # Set up a tensor for the gradients of the screen-space means.
#         means_2d = torch.zeros_like(gaussian_means, requires_grad=True)
#         try:
#             means_2d.retain_grad()
#         except Exception:
#             pass

#         settings = GaussianRasterizationSettings(
#             image_height=h,
#             image_width=w,
#             tanfovx=tan_fov_x[i].item(),
#             tanfovy=tan_fov_y[i].item(),
#             bg=background_color,
#             scale_modifier=1.0,
#             viewmatrix=view_matrix[i],
#             projmatrix=full_projection[i],
#             sh_degree=degree,
#             campos=extrinsics[i, :3, 3],
#             prefiltered=False,  # This matches the original usage.
#             render_mask=(
#                 render_masks[i]
#                 if render_masks is not None
#                 else torch.tensor([]).to(device)
#             ),
#             weight_thres=weight_thres,
#             debug=False,
#             config=torch.tensor([1.0, 1.0, 1.0, importance_config, front_config]).to(
#                 device
#             ),
#         )
#         rasterizer = GaussianRasterizer(settings)
#         rgb, normal, depth, opacity, confidence, importance, count = rasterizer(
#             means3D=gaussian_means,
#             means2D=means_2d,
#             opacities=gaussian_opacities[..., None],
#             confidences=gaussian_confidences,
#             shs=shs if use_sh else None,
#             colors_precomp=None if use_sh else shs[:, 0, :],
#             scales=gaussian_scales,
#             rotations=gaussian_rotations,
#             cov3D_precomp=None,  # gaussian_covariances[:, row, col],
#         )
#         mask = opacity.detach() > 1e-2
#         normal = torch.nn.functional.normalize(normal, dim=0) * mask
#         visible_mask = torch.sum(normal * raydir_maps[i], dim=0) < 0.0
#         confidence *= visible_mask.long()
#         # pdb.set_trace()
#         d2n = depth2normal(depth, mask, [fov_x[0], fov_y[0]])
#         all_d2ns[i] = d2n

#         # uncertainty_depth = uncertainty[:1, ...]  # depth
#         # uncertainty = 1 - torch.sum(d2n * normal, dim=0, keepdim=True)
#         # uncertainty_normal = uncertainty[1:2, ...]  # normal
#         # uncertainty = uncertainty_depth + uncertainty_normal
#         # uncertainty_rgb = uncertainty[2:, ...]  # rgb
#         # uncertainty = uncertainty_normal + uncertainty_rgb
#         # uncertainty = torch.sum(uncertainty, dim=0, keepdim=True)
#         # uncertainty = (uncertainty[0, ...] + uncertainty[2, ...]).unsqueeze(0)
#         # uncertainty[test_mask.unsqueeze(0)] = 1
#         # all_uncertainties[i] = torch.clamp(uncertainty, min=0, max=1)

#         all_rgbs[i] = rgb
#         all_normals[i] = normal
#         all_depths[i] = depth / scale[i]
#         all_opacities[i] = opacity
#         all_confidences[i] = confidence
#         all_importances[i] = importance
#         all_counts[i] = count
#     t3 = time.time()
#     print("render time:", (t2 - t1) / (t3 - t1), (t3 - t2) / (t3 - t1))

#     return (
#         all_rgbs,
#         all_depths,
#         all_normals,
#         all_opacities,
#         all_d2ns,
#         all_confidences,
#         all_importances,
#         all_counts,
#     )


def render_cuda_core(
    cam_pos,
    fov: Float[Tensor, "2"],
    view_matrix: Float[Tensor, "4 4"],
    projection_matrix: Float[Tensor, "4 4"],
    render_mask,
    raydir_map,
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "3"],
    gaussian_means: Float[Tensor, "gaussian 3"],
    gaussian_sh_coefficients: Float[Tensor, "gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "gaussian"],
    gaussian_confidences: Float[Tensor, "gaussian"],
    gaussian_scales,
    gaussian_rotations,
    front_only=False,
    require_importance: bool = False,
    use_sh: bool = False,
    weight_thres=0.03,
) -> Float[Tensor, "batch 3 height width"]:

    front_config = 0.0
    if front_only:
        front_config = 1.0

    importance_config = 0.0
    if require_importance:
        importance_config = 1.0

    device = gaussian_means.device
    tan_fov = (0.5 * fov).tan()
    means_2d = torch.zeros_like(gaussian_means, requires_grad=True)
    try:
        means_2d.retain_grad()
    except Exception:
        pass

    settings = GaussianRasterizationSettings(
        image_height=image_shape[0],
        image_width=image_shape[1],
        tanfovx=tan_fov[0].item(),
        tanfovy=tan_fov[1].item(),
        bg=background_color,
        scale_modifier=1.0,
        viewmatrix=view_matrix,
        projmatrix=projection_matrix,
        sh_degree=0,
        campos=cam_pos,
        prefiltered=False,  # This matches the original usage.
        render_mask=render_mask,
        weight_thres=weight_thres,
        debug=False,
        config=torch.tensor([1.0, 1.0, 1.0, importance_config, front_config]).to(
            device
        ),
    )
    rasterizer = GaussianRasterizer(settings)

    rgb, normal, depth, opacity, confidence, importance, count, radii = rasterizer(
        means3D=gaussian_means,
        means2D=means_2d,
        opacities=gaussian_opacities[..., None],
        confidences=gaussian_confidences,
        shs=None,
        colors_precomp=None if use_sh else gaussian_sh_coefficients[:, 0, :],
        scales=gaussian_scales,
        rotations=gaussian_rotations,
        cov3D_precomp=None,  # gaussian_covariances[:, row, col],
    )
    mask = opacity.detach() > 1e-2
    normal = torch.nn.functional.normalize(normal, dim=0) * mask
    visible_mask = torch.sum(normal * raydir_map, dim=0) < 0.0
    # confidence *= visible_mask.long()
    d2n = depth2normal(depth, mask, fov)

    return rgb, depth, normal, opacity, d2n, confidence, importance, count, radii


class GaussianRenderer:
    def __init__(
        self,
        extrinsics,
        intrinsics,
        gaussians_attr,
        background_color,
        near_far,
        resolution,
        device,
        render_masks=None,
    ):
        self.device = device
        (
            self.gaussian_means,
            self.gaussian_harmonics,
            self.gaussian_opacities,
            self.gaussian_confidences,
            self.gaussian_scales,
            self.gaussian_rotations,
        ) = gaussians_attr

        self.background_color = background_color
        self.h, self.w = resolution

        self.batch_size, _, _ = extrinsics.shape
        self.cam_pos = extrinsics[:, :3, 3]
        near = repeat(
            torch.tensor(near_far[0], device=self.device), "-> b", b=self.batch_size
        )
        far = repeat(
            torch.tensor(near_far[1], device=self.device), "-> b", b=self.batch_size
        )
        fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
        self.fovs = torch.stack([fov_x, fov_y], dim=-1)
        projection_matrices_cam = get_projection_matrix(near, far, fov_x, fov_y)
        projection_matrices_cam = rearrange(projection_matrices_cam, "b i j -> b j i")
        self.view_matrices = rearrange(extrinsics.inverse(), "b i j -> b j i")

        self.projection_matrices = self.view_matrices @ projection_matrices_cam

        xy_ray, _ = sample_image_grid((self.h, self.w), self.device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        directions = unproject(
            xy_ray,
            torch.ones_like(xy_ray[..., 0]),
            intrinsics,
        ).view(self.h, self.w, -1, 3)
        raydir_maps = torch.nn.functional.normalize(directions, dim=-1)
        self.raydir_map = raydir_maps.permute(2, 3, 0, 1)[0]  # 3 H W
        if render_masks is None:
            self.render_masks = [
                torch.tensor([], device=self.device) for _ in range(self.batch_size)
            ]
        else:
            self.render_masks = render_masks

    def update_attr(self, gaussians_attr):
        (
            self.gaussian_means,
            self.gaussian_harmonics,
            self.gaussian_opacities,
            self.gaussian_confidences,
            self.gaussian_scales,
            self.gaussian_rotations,
        ) = gaussians_attr

    # render image at a certain view
    def render_view(
        self, i=0, require_grad=False, require_importance=False, front_only=False
    ):
        with torch.set_grad_enabled(require_grad):
            (rgb, depth, normal, opacity, d2n, confidence, importance, count, radii) = (
                render_cuda_core(
                    self.cam_pos[i],
                    self.fovs[i],
                    self.view_matrices[i],
                    self.projection_matrices[i],
                    self.render_masks[i],
                    self.raydir_map,
                    (self.h, self.w),
                    self.background_color,
                    self.gaussian_means,
                    self.gaussian_harmonics,
                    self.gaussian_opacities,
                    self.gaussian_confidences,
                    self.gaussian_scales,
                    self.gaussian_rotations,
                    front_only=front_only,
                    require_importance=require_importance,
                )
            )
            in_frumstum_mask = radii > 0
        return (
            rgb,
            depth,
            normal,
            opacity,
            d2n,
            confidence,
            importance,
            count,
            in_frumstum_mask,
        )

    # render all images
    def render_view_all(
        self, require_grad=False, require_importance=False, front_only=False
    ):
        num_gaussians = len(self.gaussian_means)
        all_rgbs = torch.empty(self.batch_size, 3, self.h, self.w, device=self.device)
        all_depths = torch.empty(self.batch_size, 1, self.h, self.w, device=self.device)
        all_d2ns = torch.empty(self.batch_size, 3, self.h, self.w, device=self.device)
        all_normals = torch.empty(
            self.batch_size, 3, self.h, self.w, device=self.device
        )
        all_opacities = torch.empty(
            self.batch_size, 1, self.h, self.w, device=self.device
        )
        all_confidences = torch.empty(
            self.batch_size, 1, self.h, self.w, device=self.device
        )
        all_importances = torch.zeros(
            self.batch_size, num_gaussians, device=self.device
        )
        all_counts = torch.zeros(
            self.batch_size, num_gaussians, device=self.device, dtype=torch.int32
        )
        all_radiis = torch.zeros(num_gaussians, device=self.device, dtype=torch.int32)

        with torch.set_grad_enabled(require_grad):
            for i in range(self.batch_size):
                (
                    rgb,
                    depth,
                    normal,
                    opacity,
                    d2n,
                    confidence,
                    importance,
                    count,
                    radii,
                ) = render_cuda_core(
                    self.cam_pos[i],
                    self.fovs[i],
                    self.view_matrices[i],
                    self.projection_matrices[i],
                    self.render_masks[i],
                    self.raydir_map,
                    (self.h, self.w),
                    self.background_color,
                    self.gaussian_means,
                    self.gaussian_harmonics,
                    self.gaussian_opacities,
                    self.gaussian_confidences,
                    self.gaussian_scales,
                    self.gaussian_rotations,
                    front_only=front_only,
                    require_importance=require_importance,
                )

                all_rgbs[i] = rgb
                all_normals[i] = normal
                all_depths[i] = depth
                all_d2ns[i] = d2n
                all_opacities[i] = opacity
                all_confidences[i] = confidence
                all_importances[i] = importance
                all_counts[i] = count
                all_radiis += radii
        in_frumtum_mask = all_radiis > 0
        return (
            all_rgbs,
            all_depths,
            all_normals,
            all_opacities,
            all_d2ns,
            all_confidences,
            all_importances,
            all_counts,
            in_frumtum_mask,
        )
