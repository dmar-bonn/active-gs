import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm

from utils.operations import *
from utils.common import TextColors
from .utils import (
    l1_loss_fc_mask,
    normal_tv_loss_fc,
    cons_loss_fc,
    UniformSampler,
    WeightedSampler,
)


class GaussianMap:
    def __init__(self, cfg, device):
        self.device = device

        # trainable gaussian parameters
        self._means = torch.empty(0, device=device)
        self._scales = torch.empty(0, device=device)
        self._rotations = torch.empty(0, device=device)
        self._opacities = torch.empty(0, device=device)
        self._harmonics = torch.empty(0, device=device)

        # non-trainable gaussian parameters for confidence
        self.view_scores = torch.empty(0, device=device)
        self.view_supports = torch.empty(0, device=device)
        self.view_means = torch.empty((0, 3), device=self.device)

        self.training_performance = torch.tensor([], device=device)
        self.training_data = []
        self.is_init = False

        self.use_view_distribution = True

        # cfg is only necessary for init training
        if cfg is not None:
            self.cfg = cfg
            self.use_view_distribution = cfg.use_view_distribution
            self.scene_near, self.scene_far = cfg.bound
            self.sparse_ratio = cfg.sparse_ratio
            self.scale_factor = cfg.scale_factor
            self.error_thres = cfg.error_thres
            self.prune_interval = cfg.prune_interval
            self.optimization_steps = cfg.optimization_steps
            self.background_color = torch.tensor(
                cfg.background, dtype=torch.float32
            ).to(self.device)

        # activation function
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def update(self, dataframe):
        self.add_gaussians(dataframe)
        self.train()

    def train(self, steps=None):
        """
        train GS map for a certain steps
        """

        torch.cuda.empty_cache()
        self.init_training()
        training_sampler = self.get_sampler(self.training_data)
        iterations = self.optimization_steps if steps is None else steps

        for i in tqdm(
            range(iterations),
            desc=f" {TextColors.CYAN}Train Gaussian Map{TextColors.RESET}",
        ):
            [rgb_gts, depth_gts, extrinsics, intrinsics], frame_ids = (
                training_sampler.next_frames(self.training_performance)
            )
            *_, h, w = rgb_gts.shape
            (
                rgb_preds,
                depth_preds,
                normal_preds,
                opacity_preds,
                d2n_preds,
                _,
                _,
                _,
                _,
            ) = GaussianRenderer(
                extrinsics,
                intrinsics,
                self.get_attr(),
                self.background_color,
                (self.scene_near, self.scene_far),
                (h, w),
                self.device,
            ).render_view_all(
                require_grad=True
            )

            mask_vis = opacity_preds.detach() > 1e-3
            mask_depth = depth_gts > 0.0

            rgb_loss = l1_loss_fc_mask(rgb_preds, rgb_gts, mask_vis)
            depth_loss = l1_loss_fc_mask(depth_preds, depth_gts, mask_depth)
            self.track_performance(rgb_loss, depth_loss, frame_ids)

            rgb_loss = rgb_loss.mean()
            depth_loss = depth_loss.mean()
            normal_cons_loss = normal_tv_loss_fc(normal_preds, depth_preds, mask_depth)
            consistency_loss = cons_loss_fc(normal_preds, d2n_preds)
            consistency_loss = (consistency_loss * mask_vis.long()).mean()

            total_loss = (
                rgb_loss
                + 0.8 * depth_loss
                + 0.1 * consistency_loss
                + 0.1 * normal_cons_loss
            )
            total_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        self.post_processing()
        self.is_init = True

    def track_performance(self, rgb_loss, depth_loss, frame_ids):
        """
        track rendering performance at all keyframes
        """

        rgb_errs = torch.mean(rgb_loss, dim=[1, 2, 3])
        depth_errs = torch.mean(depth_loss, dim=[1, 2, 3])
        self.training_performance[frame_ids] = rgb_errs.detach() + depth_errs.detach()

    def post_processing(self):
        """
        update confidence and prune invisible gaussians
        """

        num_training_frame = len(self.training_data)
        require_prune = num_training_frame % self.prune_interval == 0

        # if require pruning, post process using all training views
        if require_prune:
            extrinsics = torch.stack(
                [self.training_data[i]["extrinsic"] for i in range(num_training_frame)]
            )
            intrinsics = torch.stack(
                [self.training_data[i]["intrinsic"] for i in range(num_training_frame)]
            )
            rgb_gts = torch.stack(
                [self.training_data[i]["rgb"] for i in range(num_training_frame)]
            )
            depth_gts = torch.stack(
                [self.training_data[i]["depth"] for i in range(num_training_frame)]
            )

        # else only use the latest view for confidence update
        else:
            extrinsics = self.training_data[-1]["extrinsic"].unsqueeze(0)
            intrinsics = self.training_data[-1]["intrinsic"].unsqueeze(0)
            rgb_gts = self.training_data[-1]["rgb"].unsqueeze(0)
            depth_gts = self.training_data[-1]["depth"].unsqueeze(0)

        depth_ranges = self.training_data[-1]["depth_range"]
        *_, h, w = rgb_gts.shape
        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            counts,
            _,
        ) = GaussianRenderer(
            extrinsics,
            intrinsics,
            self.get_attr(),
            self.background_color,
            (self.scene_near, self.scene_far),
            (h, w),
            self.device,
            render_masks=(depth_gts > 0.0).float(),
        ).render_view_all(require_importance=True, front_only=True)

        # update visible gaussian info
        update_mask = counts[-1] >= 1.0  # visible gaussians for the latest view
        self.view_supports += update_mask.float()  # total visible views for a gaussian

        if self.use_view_distribution:
            # update confidence
            gaussian_means = self.get_means.detach()
            gaussian_normals = self.get_normals.detach()
            view_directions = extrinsics[-1:, :3, 3] - gaussian_means
            view_distances = torch.linalg.norm(
                view_directions,
                dim=1,
            )
            view_directions = view_directions / view_distances.unsqueeze(
                -1
            )  # normalize
            delta = view_directions[update_mask] - self.view_means[update_mask]

            # view mean vectors
            self.view_means[update_mask] += delta / self.view_supports[
                update_mask
            ].unsqueeze(-1)

            # view scores
            cosine_sim = torch.clamp(
                torch.sum(gaussian_normals * view_directions, 1), min=0, max=1
            )
            distance_factor = torch.clamp(
                view_distances / depth_ranges[1], min=0, max=1
            )
            self.view_scores[update_mask] += (1 - distance_factor)[
                update_mask
            ] * cosine_sim[update_mask]

        if require_prune:
            vis_mask = (
                torch.sum(counts, dim=0) >= 1.0
            )  # visible to at least one training view
            self.prune(~vis_mask)

    def prune(self, prune_mask):
        prune_mask += self.get_opacities < 0.1
        prune_mask = prune_mask.bool()
        self._means = self._means[~prune_mask]
        self._scales = self._scales[~prune_mask]
        self._rotations = self._rotations[~prune_mask]
        self._opacities = self._opacities[~prune_mask]
        self._harmonics = self._harmonics[~prune_mask]
        self.view_scores = self.view_scores[~prune_mask]
        self.view_supports = self.view_supports[~prune_mask]
        self.view_means = self.view_means[~prune_mask]

        print(f"delete {torch.sum(prune_mask)} gaussians")

    def get_sampler(self, training_data):
        """
        get training data sampler
        """

        if self.cfg.sampler.sampler_type == "uniform":
            sampler = UniformSampler(self.cfg.sampler, training_data)
        elif self.cfg.sampler.sampler_type == "weighted":
            sampler = WeightedSampler(self.cfg.sampler, training_data)
        return sampler

    def init_training(self):
        self._means = nn.Parameter(self._means)
        self._scales = nn.Parameter(self._scales)
        self._rotations = nn.Parameter(self._rotations)
        self._opacities = nn.Parameter(self._opacities)
        self._harmonics = nn.Parameter(self._harmonics)
        l = [
            {
                "params": [self._means],
                "lr": self.cfg.optimizer.mean_lr,
                "name": "mean",
            },
            {
                "params": [self._scales],
                "lr": self.cfg.optimizer.scale_lr,
                "name": "scale",
            },
            {
                "params": [self._rotations],
                "lr": self.cfg.optimizer.rotation_lr,
                "name": "rotation",
            },
            {
                "params": [self._opacities],
                "lr": self.cfg.optimizer.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._harmonics],
                "lr": self.cfg.optimizer.harmonic_lr,
                "name": "harmonic",
            },
        ]
        self.optimizer = torch.optim.Adam(l, eps=1e-15)

    def add_gaussians(self, dataframe):
        rgb = dataframe["rgb"]
        depth = dataframe["depth"]
        depth_smooth = get_smooth_depth(depth.squeeze(0).cpu().numpy())
        depth_smooth = torch.tensor(depth_smooth, device=self.device).unsqueeze(0)
        intrinsic = dataframe["intrinsic"]
        extrinsic = dataframe["extrinsic"]
        valid_mask = (depth > 0.0).view(-1)

        _, H, W = rgb.shape
        point_num = H * W
        xy_ray, _ = sample_image_grid((H, W), device=self.device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        origins, directions = get_world_rays(xy_ray, extrinsic, intrinsic)
        pcd = (origins + directions * depth.view(-1, 1, 1)).squeeze(1)  # (H*W, 3)

        pcd_normals = torch.zeros(point_num, 3, device=self.device)
        pcd_normals[:, 2] = 1.0
        pcd_normals_cam = torch.zeros(point_num, 3, device=self.device)
        pcd_normals_cam[:, 2] = 1.0

        # use depth map to generate normal
        normals_cam = (
            depth2normal(
                depth_smooth, valid_mask.view(1, H, W), fov=(np.pi / 3, np.pi / 3)
            )
            .permute(1, 2, 0)
            .view(-1, 3)
        )
        valid_normal_mask = torch.sum(normals_cam**2, dim=-1) > 0.0
        valid_mask *= valid_normal_mask

        normals_world = torch.matmul(extrinsic[:3, :3], normals_cam.T).T
        pcd_normals_cam[valid_mask] = normals_cam[valid_mask]
        pcd_normals[valid_mask] = normals_world[valid_mask]

        # remove normals that are non-visible
        directions_norm = torch.nn.functional.normalize(
            directions.squeeze(1), dim=1
        )  # N, 3
        cos_sim = torch.sum(directions_norm * pcd_normals, dim=-1)
        valid_normal_mask = cos_sim < -0.01
        valid_mask *= valid_normal_mask

        if self.is_init:
            (
                rgb_pred,
                depth_pred,
                normal_pred,
                opacity_pred,
                _,
                confidence_pred,
                _,
                _,
                _,
            ) = GaussianRenderer(
                extrinsic.unsqueeze(0).to(self.device),
                intrinsic.unsqueeze(0).to(self.device),
                self.get_attr(),
                self.background_color,
                (self.scene_near, self.scene_far),
                (H, W),
                self.device,
            ).render_view_all()

            global_render_results = {
                "rgb": rgb_pred,
                "depth": depth_pred.squeeze(1),
                "opacity": opacity_pred.squeeze(1),
                "confidence": confidence_pred.squeeze(1),
                "normal": normal_pred,
            }

        else:
            global_render_results = None

        means_new = pcd
        rotations_new, _ = normal2rotation(pcd_normals)
        scales_new = torch.zeros_like(means_new, device=self.device)
        scales_new[:, -1] -= 1e10
        opacities_new = torch.zeros(point_num, device=self.device)
        harmonics_new = torch.zeros(point_num, 1, 3, device=self.device)
        harmonics_new[:, 0, :] = rgb.permute(1, 2, 0).view(-1, 3)

        # non-learnable parameters
        view_scores_new = torch.zeros(point_num, device=self.device)
        view_supports_new = torch.zeros(point_num, device=self.device)
        view_means_new = torch.zeros((point_num, 3), device=self.device)

        #############################
        # rotaion_nan = torch.sum(torch.isnan(rotations_new))
        # if rotaion_nan > 0:
        #     print("has nan in rotation new")
        #     pdb.set_trace()
        #############################
        nan_rotation_mask = torch.any(rotations_new.isnan(), dim=1)
        valid_mask *= ~nan_rotation_mask

        select_mask = self.cal_mask(
            rgb.unsqueeze(0),
            depth.unsqueeze(0),
            global_render_results,
        )
        select_mask = select_mask.to(self.device).squeeze(0) * valid_mask
        selected_idx = torch.nonzero(select_mask, as_tuple=False).flatten()

        # voxel filtering
        select_mask_final = torch.zeros_like(select_mask, dtype=torch.bool)
        test_mask = torch.zeros(len(selected_idx), dtype=torch.bool)
        selected_pcd = pcd[select_mask]
        vf_idx = voxel_downsample(selected_pcd.to(self.device))
        test_mask[vf_idx] = True
        selected_idx = selected_idx[test_mask]
        select_mask_final[selected_idx] = True
        select_mask = select_mask_final

        self._means = torch.cat(
            (self._means.detach(), means_new.float()[select_mask]),
            dim=0,
        )
        self._scales = torch.cat(
            (self._scales.detach(), scales_new.float()[select_mask]),
            dim=0,
        )

        self._harmonics = torch.cat(
            (
                self._harmonics.detach(),
                harmonics_new.float()[select_mask],
            ),
            dim=0,
        )
        self._opacities = torch.cat(
            (
                self._opacities.detach(),
                opacities_new.float()[select_mask],
            ),
            dim=0,
        )
        self._rotations = torch.cat(
            (
                self._rotations.detach(),
                rotations_new.float()[select_mask],
            ),
            dim=0,
        )

        self.view_scores = torch.cat(
            (
                self.view_scores,
                view_scores_new.float()[select_mask],
            ),
            dim=0,
        )

        self.view_supports = torch.cat(
            (
                self.view_supports,
                view_supports_new.float()[select_mask],
            ),
            dim=0,
        )

        self.view_means = torch.cat(
            (
                self.view_means,
                view_means_new.float()[select_mask],
            ),
            dim=0,
        )

        self.training_data.append(dataframe)
        self.training_performance = torch.cat(
            (self.training_performance, torch.tensor([10], device=self.device)), 0
        )

    def cal_mask(self, rgb_gt, depth_gt, pred):
        """
        get mask for spawning new gaussian primitives
        """

        v, _, h, w = rgb_gt.shape
        device = rgb_gt.device
        if pred is not None:
            rgb = pred["rgb"].to(device)
            depth = pred["depth"].to(device)
            opacity = pred["opacity"].to(device)

            rgb_error = torch.mean((rgb_gt - rgb) ** 2, dim=1)
            mask = rgb_error > self.error_thres
            mask += opacity < 0.5
            mask += (depth_gt.squeeze(0) - depth) < -0.05 * depth_gt.squeeze(0)
        else:
            mask = torch.ones(v, h, w).to(device)

        return rearrange(mask.bool(), "v h w -> (v h w)")

    def save(self, save_path, index="final"):
        map_state = {
            "means": self._means.detach(),
            "scales": self._scales.detach(),
            "harmonics": self._harmonics.detach(),
            "opacities": self._opacities.detach(),
            "rotations": self._rotations.detach(),
            "view_scores": self.view_scores.detach(),
            "view_supports": self.view_supports.detach(),
            "view_means": self.view_means.detach(),
            "near": self.scene_near,
            "far": self.scene_far,
            "use_view_direction": self.use_view_distribution,
            "background_color": self.background_color,
            "scale_factor": self.scale_factor,
        }
        torch.save(map_state, f"{save_path}/map_{index}.th")

    def load(self, model_path):
        map_state = torch.load(model_path)
        # gaussian map state
        self._means = map_state["means"]
        self._scales = map_state["scales"]
        self._harmonics = map_state["harmonics"]
        self._opacities = map_state["opacities"]
        self._rotations = map_state["rotations"]
        self.view_scores = map_state["view_scores"]
        self.view_supports = map_state["view_supports"]
        self.view_means = map_state["view_means"]
        self.scene_near = map_state["near"]
        self.scene_far = map_state["far"]
        # self.use_view_distribution = map_state["use_view_direction"]
        self.background_color = torch.tensor(
            map_state["background_color"], dtype=torch.float32
        ).to(self.device)
        self.scale_factor = map_state["scale_factor"]
        self.is_init = True

    @property
    def get_means(self):
        return self._means

    @property
    def get_rotations(self):
        return self.rotation_activation(self._rotations)

    @property
    def get_scales(self):
        return torch.clamp(
            self.scale_factor * self.scaling_activation(self._scales), min=0, max=0.05
        )

    @property
    def get_opacities(self):
        return self.opacity_activation(self._opacities)

    @property
    def get_harmonics(self):
        return self._harmonics

    @property
    def get_confidences(self):
        if self.use_view_distribution:
            view_var = self.view_means.norm(dim=-1)
            view_var[torch.isnan(view_var)] = 1.0
            view_variance_factor = torch.exp(1 - view_var)
            confidences = torch.clamp(
                view_variance_factor * self.view_scores, min=0, max=1
            )
        else:
            confidences = torch.clamp(
                1 - 1 / torch.exp(self.view_supports), min=0, max=1
            )

        return confidences

    @property
    def get_normals(self):
        return self.rotation_activation(
            quaternion_to_matrix(self.get_rotations)[:, :3, 2]
        )

    def get_attr(self):
        return (
            self.get_means,
            self.get_harmonics,
            self.get_opacities,
            self.get_confidences,
            self.get_scales,
            self.get_rotations,
        )

    def get_params(self):
        return (
            self._means,
            self._harmonics,
            self._opacities,
            self._scales,
            self._rotations,
        )
