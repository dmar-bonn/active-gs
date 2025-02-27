from .plan_base import *
from utils.common import TextColors


class Random(PlanBase):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)

    @torch.no_grad
    def cal_utility(self, gaussian_map, voxel_map, candidates, simulator):

        print(f" {TextColors.CYAN}Evaluate View Candidates{TextColors.RESET}")
        utility_list = torch.rand(len(candidates))
        return utility_list, 0
