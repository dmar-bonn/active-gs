import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
import json
import cv2
import pdb
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import os
import torchmetrics

# from utils.operations import render_cuda, build_covariance
from tqdm import tqdm
import math
from dataclasses import dataclass
from jaxtyping import Float
from torch import Tensor


@dataclass
class Gaussians:
    means: Float[Tensor, "*batch 3"]
    scales: Float[Tensor, "*batch 3"]
    rotations: Float[Tensor, "*batch 4"]
    harmonics: Float[Tensor, "*batch 3 _"]
    opacities: Float[Tensor, " *batch"]
    variances: Float[Tensor, " *batch"]


def _tensor_size(t):
    return t.size()[1] * t.size()[2]


def cons_loss_fc(normals, depth_normals):
    cos = torch.sum(normals * depth_normals, 1)
    return 1 - cos


def geo_loss_fc(normals):
    b, _, h, w = normals.shape
    count_h = _tensor_size(normals[:, :, 1:, :])
    count_w = _tensor_size(normals[:, :, :, 1:])
    h_tv = torch.pow((normals[:, :, 1:, :] - normals[:, :, : h - 1, :]), 2).sum()
    w_tv = torch.pow((normals[:, :, :, 1:] - normals[:, :, :, : w - 1]), 2).sum()
    return 2 * (h_tv / count_h + w_tv / count_w) / b


def normal_tv_loss_fc(normals, depths, mask, sigma=0.3):
    # Compute edge-aware weights
    normal_diff_norms = central_diff(normals)
    depth_diff_norms = central_diff(depths.detach())

    depth_mask = (depth_diff_norms <= 0.0001).float()
    weights = torch.exp(-normal_diff_norms / (2 * sigma**2))  # Shape: (b, 4, h, w)

    # Compute weighted normal consistency loss
    loss = torch.mean(depth_mask * weights * normal_diff_norms * mask)
    # loss = torch.mean(normal_diff_norms)
    return loss


def central_diff(map):
    shift_left = map[:, :, :, :-1] - map[:, :, :, 1:]
    shift_right = map[:, :, :, 1:] - map[:, :, :, :-1]
    shift_up = map[:, :, :-1, :] - map[:, :, 1:, :]
    shift_down = map[:, :, 1:, :] - map[:, :, :-1, :]

    pad = (0, 1, 0, 0)  # Padding for left-shifted differences
    shift_left = F.pad(shift_left, pad, mode="constant", value=0)
    pad = (1, 0, 0, 0)  # Padding for right-shifted differences
    shift_right = F.pad(shift_right, pad, mode="constant", value=0)
    pad = (0, 0, 0, 1)  # Padding for up-shifted differences
    shift_up = F.pad(shift_up, pad, mode="constant", value=0)
    pad = (0, 0, 1, 0)  # Padding for down-shifted differences
    shift_down = F.pad(shift_down, pad, mode="constant", value=0)
    diffs = torch.stack(
        [shift_left, shift_right, shift_up, shift_down], dim=2
    )  # Shape: (b, 3, 4, h, w)

    # Compute the squared norm of the differences
    diff_norms = torch.sum(diffs**2, dim=1)  # Shape: (b, 4, h, w)
    return diff_norms


def normal_reg_fc(normals, masks):
    # normals shape: (n, 3, h, w)
    n, c, h, w = normals.shape

    # Padding to handle the borders
    normals_padded = F.pad(normals, (1, 1, 1, 1), mode="replicate")
    # Unfold the padded tensor to get 3x3 neighborhoods
    neighborhoods = normals_padded.unfold(2, 3, 1).unfold(3, 3, 1)
    # neighborhoods shape: (n, 3, h, w, 3, 3)

    # Reshape neighborhoods to get all 8 neighbors and the central pixel
    neighbors = neighborhoods.permute(0, 2, 3, 4, 5, 1).reshape(n, h, w, 3, -1)
    # neighbors shape: (n, h, w, 3, 9)

    # Separate the central pixel from the neighbors
    central_pixel = neighbors[:, :, :, :, 4]
    neighbors = torch.cat(
        [neighbors[:, :, :, :, :4], neighbors[:, :, :, :, 5:]], dim=-1
    )
    # central_pixel shape: (n, h, w, 3)
    # neighbors shape: (n, h, w, 3, 8)

    # Compute dot product
    # dot_product = torch.einsum("nhwc,nhwkc->nhwk", central_pixel, neighbors)
    dot_product = (central_pixel.unsqueeze(-1) * neighbors).sum(dim=-2)
    # dot_product shape: (n, h, w, 8)

    # Compute norms
    central_norm = torch.norm(central_pixel, dim=-1, keepdim=True)
    neighbor_norm = torch.norm(neighbors, dim=-2)
    # central_norm shape: (n, h, w, 1)
    # neighbor_norm shape: (n, h, w, 8)

    # Compute cosine similarity
    cosine_similarity = dot_product / (central_norm * neighbor_norm + 1e-8)
    loss = torch.mean(1 - cosine_similarity, dim=-1)
    return (loss * masks).mean()


def scale_loss_fc(scales):
    scale_mean = torch.mean(scales[..., :2], dim=-1, keepdim=True)
    scale_diff = torch.mean(torch.abs(scales[..., :2] - scale_mean))
    scale_loss = torch.mean(scale_diff)
    # max_scale, _ = torch.max(scales, dim=-1)
    # min_scale, _ = torch.min(scales, dim=-1)
    # scale_ratio = min_scale / (max_scale + 1.0e-8)
    # scale_loss = torch.mean(1 - scale_ratio)
    return scale_loss


# def op_loss_fc(opacities):
#     op_loss = torch.mean(1 - opacities)
#     return op_loss


def op_loss_fc(opacities, confidences):
    loss = (1 - confidences) * opacities
    return loss.mean()


# def op_loss_fc(opacities):
#     op_loss = torch.exp(-((opacities - 0.5) ** 2) / 0.05).mean()
#     return op_loss


def l1_loss_fc(network_output, gt):
    # valid_mask = gt > 0.0
    return torch.abs((network_output - gt)).mean()


def l1_loss_fc_mask(network_output, gt, mask):
    return torch.abs((network_output - gt) * mask)


def l2_loss_fc(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim_loss_fc(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class WeightedSampler:
    def __init__(self, cfg, dataframes):
        active_size = min(cfg.active_size, len(dataframes))
        batch_size = cfg.batch_size
        self.dataframes = dataframes
        ids = range(len(self.dataframes))
        self.random_num = batch_size - active_size
        # assert len(ids) >= active_size

        self.active_ids = np.array(ids[-active_size:])
        self.random_ids_all = np.array(ids[:-active_size])
        self.selected_num = min(len(self.random_ids_all), self.random_num)
        self.v = (
            len(self.active_ids) + self.selected_num
        )  # total frame num used for training

    def next_frames(self, weight):
        selected_ids = self.active_ids.copy()
        if self.selected_num > 0:
            weight = weight[self.random_ids_all]
            weight /= torch.sum(weight)
            indices = np.random.choice(
                self.random_ids_all,
                size=self.selected_num,
                p=weight.cpu().numpy(),
                replace=False,
            )
            ids = self.random_ids_all[indices]
            selected_ids = np.append(selected_ids, ids)

        rgbs = torch.stack([self.dataframes[i]["rgb"] for i in selected_ids])
        depths = torch.stack([self.dataframes[i]["depth"] for i in selected_ids])
        extrinsics = torch.stack(
            [self.dataframes[i]["extrinsic"] for i in selected_ids]
        )
        intrinsics = torch.stack(
            [self.dataframes[i]["intrinsic"] for i in selected_ids]
        )
        return [rgbs, depths, extrinsics, intrinsics], selected_ids


class UniformSampler:
    def __init__(self, cfg, dataframes):
        self.dataframes = dataframes
        ids = list(self.dataframes.keys())
        self.random_num = cfg.batch_size - cfg.active_size
        assert len(ids) >= cfg.active_size

        self.active_ids = np.array(ids[-cfg.active_size :])
        self.random_ids_all = np.array(ids[: -cfg.active_size])
        self.selected_num = min(len(self.random_ids_all), self.random_num)
        self.v = (
            len(self.active_ids) + self.selected_num
        )  # total frame num used for training

    def next_frames(
        self,
    ):
        selected_ids = self.active_ids.copy()
        if self.selected_num > 0:
            indices = torch.randperm(len(self.random_ids_all))[: self.selected_num]
            ids = self.random_ids_all[indices.numpy()]
            selected_ids = np.append(selected_ids, ids)
        rgbs = torch.stack([self.dataframes[i]["rgb"] for i in selected_ids])
        depths = torch.stack([self.dataframes[i]["depth"] for i in selected_ids])
        extrinsics = torch.stack(
            [self.dataframes[i]["extrinsic"] for i in selected_ids]
        )
        intrinsics = torch.stack(
            [self.dataframes[i]["intrinsic"] for i in selected_ids]
        )
        return rgbs, depths, extrinsics, intrinsics


lpips_cal = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(
    "cuda"
)


def cal_mse(pred, gt, mask=1.0):
    mse = (((pred - gt) * mask) ** 2).mean().cpu()
    return mse


def cal_psnr(rgb_pred, rgb_gt):
    mse = cal_mse(rgb_pred, rgb_gt)
    psnr = -10 * math.log10(mse + 1e-8)
    return psnr


def cal_ssim(rgb_pred, rgb_gt):
    ssim_cal = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0)
    ssim = ssim_cal(rgb_pred.cpu(), rgb_gt.cpu()).item()
    return ssim


def cal_lpips(rgb_pred, rgb_gt):
    lpips = lpips_cal(rgb_pred, rgb_gt).item()
    return lpips

    # @torch.no_grad()
    # def eval_rendering(gaussians, dataset, kf_indices, cfg, device, save_dir=None):
    #     interval = 5

    #     background_color = torch.tensor(
    #         torch.tensor([0.0, 0.0, 0.0]), dtype=torch.float32
    #     ).to(device)
    #     near, far = cfg.mapper.bound
    #     img_pred, img_gt, test_frame_idx = [], [], []
    #     end_idx = len(dataset)
    #     psnr_array, ssim_array, lpips_array = [], [], []
    #     covariances = build_covariance(gaussians.scales, gaussians.rotations)

    #     for idx in range(0, end_idx, interval):
    #         if idx in kf_indices:
    #             continue
    #         test_frame_idx.append(idx)

    #     for idx in tqdm(test_frame_idx):
    #         dataframe = dataset[idx]
    #         rgb_gt = dataframe["rgb"]
    #         depth_gt = dataframe["depth"]
    #         extrinsic = dataframe["extrinsic"]
    #         intrinsic = dataframe["intrinsic"]

    #         rgb, depth, normal, opacity, uncertainty, importance_score, _ = render_cuda(
    #             extrinsic.unsqueeze(0).to(device),
    #             intrinsic.unsqueeze(0).to(device),
    #             torch.tensor([near]).to(device),
    #             torch.tensor([far]).to(device),
    #             cfg.mapper.resolution,
    #             background_color,
    #             gaussians.means,
    #             covariances,
    #             gaussians.harmonics,
    #             gaussians.opacities,
    #             gaussians.variances,
    #         )
    #         rgb_pred = torch.clamp(rgb.detach(), 0.0, 1.0)
    #         rgb_gt = rgb_gt.unsqueeze(0)

    #         psnr_score = cal_psnr(rgb_pred, rgb_gt)
    #         ssim_score = cal_ssim(rgb_pred, rgb_gt)
    #         lpips_score = cal_lpips(rgb_pred, rgb_gt)

    #         psnr_array.append(psnr_score)
    #         ssim_array.append(ssim_score)
    #         lpips_array.append(lpips_score)

    #     output = dict()
    #     output["mean_psnr"] = float(np.mean(psnr_array))
    #     output["mean_ssim"] = float(np.mean(ssim_array))
    #     output["mean_lpips"] = float(np.mean(lpips_array))
    #     print(output)

    # Log(
    #     f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_ssim"]}',
    #     tag="Eval",
    # )

    # psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    # mkdir_p(psnr_save_dir)

    # if save_dir is not None:
    #     json.dump(
    #         output,
    #         open(os.path.join(save_dir, "final_result.json"), "w", encoding="utf-8"),
    #         indent=4,
    #     )
    # return output


def cal_distance(points, origin):
    return torch.sqrt(torch.sum((points - origin) ** 2, dim=-1) + 1e-8)


def init_uncertainty(depth, normal):
    depth_factor = torch.exp(depth) * 0.001
    normal_factor = (
        1 + F.cosine_similarity(normal.view(-1, 3), torch.tensor([[0, 0, 1]])) * 0.01
    )
    # pdb.set_trace()
    return depth_factor.view(-1) + normal_factor - 1


def bresenham_3d_batch(starting_voxels, ending_voxels):
    x1, y1, z1 = starting_voxels[:, 0], starting_voxels[:, 1], starting_voxels[:, 2]
    x2, y2, z2 = ending_voxels[:, 0], ending_voxels[:, 1], ending_voxels[:, 2]

    # Calculate deltas
    dx = torch.abs(x2 - x1)
    dy = torch.abs(y2 - y1)
    dz = torch.abs(z2 - z1)

    # Determine the step directions
    sx = torch.sign(x2 - x1)
    sy = torch.sign(y2 - y1)
    sz = torch.sign(z2 - z1)

    # Initialize lists to collect all points
    points_x, points_y, points_z = [], [], []

    # Initialize error terms
    err1 = torch.zeros_like(dx)
    err2 = torch.zeros_like(dx)

    # Select the dominant direction to iterate over
    cond1 = (dx >= dy) & (dx >= dz)
    cond2 = (dy >= dx) & (dy >= dz)

    while True:
        # Collect the current points
        points_x.append(x1)
        points_y.append(y1)
        points_z.append(z1)

        # Identify the indices where lines are completed
        completed = (x1 == x2) & (y1 == y2) & (z1 == z2)
        if torch.all(completed):
            break

        # Update points based on which condition they satisfy
        err1[cond1] += 2 * dy[cond1]
        err2[cond1] += 2 * dz[cond1]
        x1[cond1] += sx[cond1]
        y1[cond1][err1[cond1] > dx[cond1]] += sy[cond1][err1[cond1] > dx[cond1]]
        err1[cond1][err1[cond1] > dx[cond1]] -= 2 * dx[cond1][err1[cond1] > dx[cond1]]
        z1[cond1][err2[cond1] > dx[cond1]] += sz[cond1][err2[cond1] > dx[cond1]]
        err2[cond1][err2[cond1] > dx[cond1]] -= 2 * dx[cond1][err2[cond1] > dx[cond1]]

        err1[cond2] += 2 * dx[cond2]
        err2[cond2] += 2 * dz[cond2]
        y1[cond2] += sy[cond2]
        x1[cond2][err1[cond2] > dy[cond2]] += sx[cond2][err1[cond2] > dy[cond2]]
        err1[cond2][err1[cond2] > dy[cond2]] -= 2 * dy[cond2][err1[cond2] > dy[cond2]]
        z1[cond2][err2[cond2] > dy[cond2]] += sz[cond2][err2[cond2] > dy[cond2]]
        err2[cond2][err2[cond2] > dy[cond2]] -= 2 * dy[cond2][err2[cond2] > dy[cond2]]

        err1[~cond1 & ~cond2] += 2 * dy[~cond1 & ~cond2]
        err2[~cond1 & ~cond2] += 2 * dx[~cond1 & ~cond2]
        z1[~cond1 & ~cond2] += sz[~cond1 & ~cond2]
        y1[~cond1 & ~cond2][err1[~cond1 & ~cond2] > dz[~cond1 & ~cond2]] += sy[
            ~cond1 & ~cond2
        ][err1[~cond1 & ~cond2] > dz[~cond1 & ~cond2]]
        err1[~cond1 & ~cond2][err1[~cond1 & ~cond2] > dz[~cond1 & ~cond2]] -= (
            2 * dz[~cond1 & ~cond2][err1[~cond1 & ~cond2] > dz[~cond1 & ~cond2]]
        )
        x1[~cond1 & ~cond2][err2[~cond1 & ~cond2] > dz[~cond1 & ~cond2]] += sx[
            ~cond1 & ~cond2
        ][err2[~cond1 & ~cond2] > dz[~cond1 & ~cond2]]
        err2[~cond1 & ~cond2][err2[~cond1 & ~cond2] > dz[~cond1 & ~cond2]] -= (
            2 * dz[~cond1 & ~cond2][err2[~cond1 & ~cond2] > dz[~cond1 & ~cond2]]
        )

    return torch.stack(
        [torch.stack(points_x), torch.stack(points_y), torch.stack(points_z)], dim=-1
    )
