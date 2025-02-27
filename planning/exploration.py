from .plan_base import *
from utils.common import TextColors
from utils.operations import GaussianRenderer


class Exploration(PlanBase):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.render_ratio = cfg.render_ratio

    @torch.no_grad
    def cal_utility(self, gaussian_map, voxel_map, candidates, simulator):
        t_utility = 0
        render_resolution = np.round(self.render_ratio * simulator.resolution).astype(
            int
        )
        h, w = render_resolution
        depth_range = simulator.depth_range
        extrinsics = candidates.to(self.device)
        intrinsics = repeat(simulator.intrinsic, " h w -> v h w", v=len(candidates)).to(
            self.device
        )

        renderer = GaussianRenderer(
            extrinsics,
            intrinsics,
            gaussian_map.get_attr(),
            gaussian_map.background_color,
            (gaussian_map.scene_near, gaussian_map.scene_far),
            (h, w),
            self.device,
        )
        explore_util = torch.zeros(len(candidates))

        require_valid_mask = simulator.has_missing_surface

        for i in tqdm(
            range(len(extrinsics)),
            desc=f" {TextColors.CYAN}Evaluate View Candidates{TextColors.RESET}",
        ):
            t_utility_start = time.time()
            (rgb, depth, normal, opacity, d2n, confidence, importance, count, _) = (
                renderer.render_view(i)
            )
            confidences = confidence[0]
            depths = depth[0]

            # due to missing surfaces issue in dataset,
            # we use simulator to get a valid mask at view candidates
            # to ignore the value at missing surfaces.
            if require_valid_mask:
                t_simulator_start = time.time()
                valid_mask = simulator.simulate(
                    extrinsics[i].cpu(), valid_mask_only=True
                )
                valid_mask = cv2.resize(
                    valid_mask.astype(np.uint8),
                    render_resolution,
                    interpolation=cv2.INTER_NEAREST,
                )
                valid_mask = torch.tensor(valid_mask).bool()
                t_simulator = time.time() - t_simulator_start
            else:
                valid_mask = torch.ones(*render_resolution).bool()
                t_simulator = 0

            # exploration utility
            depth_voxel = depths.clone()
            depth_voxel[depth_voxel < 0.001] = 10000  # unseen surfaces
            depth_voxel = torch.clamp(
                depth_voxel, min=depth_range[0], max=depth_range[1]
            )
            depth_voxel[~valid_mask] = -1.0
            visible_mask = voxel_map.cal_visible_mask(
                extrinsics[i],
                intrinsics[i],
                depth_voxel,
            )

            unexp_mask = voxel_map.unexplored_mask
            visible_unexp_mask = visible_mask & unexp_mask
            explore_util[i] = torch.sum(visible_unexp_mask) / len(
                voxel_map.voxel_centers
            )

            t_utility += time.time() - t_utility_start - t_simulator

        explore_util[torch.isnan(explore_util)] = 0.0

        utility = explore_util.cpu()
        return utility, t_utility
