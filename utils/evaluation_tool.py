import os
import numpy as np
import torch
import glob
import json
from PIL import Image
import trimesh
import torchvision.transforms as tf
from tqdm import tqdm
import pdb
from mapping.utils import cal_psnr, cal_ssim, cal_lpips, cal_mse
from .operations import calc_3d_mesh_metric, open3dmesh_2_trimesh, GaussianRenderer
import open3d as o3d

to_tensor = tf.ToTensor()


class EvaluationTool:
    def __init__(
        self,
        gaussian_map_list,
        mesh_list,
        test_folder,
        data_folder,
        eval_mode,
        device,
        simulator=None,
    ):

        self.gaussian_map_list = gaussian_map_list
        self.mesh_list = mesh_list
        self.eval_mode = eval_mode

        self.num_map = len(self.gaussian_map_list)

        self.data_folder = data_folder
        self.device = device

        pose_file = test_folder + "/traj.txt"
        assert os.path.exists(pose_file)

        poses_all = []
        with open(pose_file, "r") as f:
            lines = f.readlines()
        for line in lines:
            pose = np.array(list(map(float, line.split())))
            pose = torch.from_numpy(pose).float()
            poses_all.append(pose.view(4, 4))
        self.test_poses = torch.stack(poses_all)

        if simulator is not None:
            print("\n----------use online measurements from simulator----------")
            self.simulator = simulator
            self.mesh_gt = self.simulator.mesh
        else:
            print("\n----------use pre-recorded measurements----------")
            rgb_folder = test_folder + "/rgb"
            assert os.path.exists(rgb_folder)
            depth_folder = test_folder + "/depth"
            assert os.path.exists(depth_folder)
            self.rgb_paths = np.array(sorted(glob.glob(f"{rgb_folder}/*.png")))
            self.depth_paths = np.array(sorted(glob.glob(f"{depth_folder}/*.png")))
            assert len(self.rgb_paths) == len(self.depth_paths) == len(self.test_poses)
            intrinsic_file = test_folder + "/intrinsic.txt"
            assert os.path.exists(intrinsic_file)
            with open(intrinsic_file, "r") as f:
                lines = f.readlines()
            self.intrinsic = torch.tensor(lines).view(3, 3)
            self.simulator = None
            mesh_file = test_folder + "/mesh.ply"
            assert os.path.exists(mesh_file)
            self.mesh_gt = trimesh.load(mesh_file)

    def get_test_data(self, i, pose):
        if self.simulator is not None:
            dataframe = self.simulator.simulate(pose, require_gt=True)
        else:
            rgb = to_tensor((Image.open(self.rgb_paths[i])))

            ###### place holder, need scale for recovering depth
            depth = torch.from_numpy(
                np.array(Image.open(self.depth_paths[i]))
            ).unsqueeze(0)
            dataframe = {
                "rgb": rgb,
                "depth": depth,
                "extrinsic": pose,
                "intrinsic": self.intrinsic,
            }
        return dataframe

    @torch.no_grad()
    def eval(self):
        psnr = np.zeros(self.num_map)
        ssim = np.zeros(self.num_map)
        lpips = np.zeros(self.num_map)
        depth_mse = np.zeros(self.num_map)
        accuracy = np.zeros(self.num_map)
        completion = np.zeros(self.num_map)
        completion_ratio = np.zeros(self.num_map)
        chamfer_distance = np.zeros(self.num_map)

        output = dict()

        test_num = len(self.test_poses)
        idx = range(test_num)
        if self.eval_mode in ["complete", "rendering"]:
            print("evaluate rendering quality \n")
            for i in tqdm(idx):
                pose = self.test_poses[i]
                dataframe = self.get_test_data(i, pose)
                rgb_gt = dataframe["rgb"].to(self.device)
                rgb_gt = rgb_gt.unsqueeze(0)
                depth_gt = dataframe["depth"].to(self.device)
                valid_mask = (depth_gt > 0).float()
                extrinsic = dataframe["extrinsic"].to(self.device)
                intrinsic = dataframe["intrinsic"].to(self.device)

                _, _, H, W = rgb_gt.shape
                for i, gaussian_map in enumerate(self.gaussian_map_list):

                    rgb, depth, _, _, _, _, _, _, _ = GaussianRenderer(
                        extrinsic.unsqueeze(0),
                        intrinsic.unsqueeze(0),
                        gaussian_map.get_attr(),
                        gaussian_map.background_color,
                        (gaussian_map.scene_near, gaussian_map.scene_far),
                        (H, W),
                        self.device,
                    ).render_view_all()
                    rgb_pred = torch.clamp(rgb, 0.0, 1.0)

                    psnr_score = cal_psnr(rgb_pred, rgb_gt)
                    ssim_score = cal_ssim(rgb_pred, rgb_gt)
                    lpips_score = cal_lpips(rgb_pred, rgb_gt)
                    depth_mse_score = cal_mse(depth, depth_gt, valid_mask)

                    psnr[i] += psnr_score
                    ssim[i] += ssim_score
                    lpips[i] += lpips_score
                    depth_mse[i] += depth_mse_score

            output["mean_psnr"] = (psnr / test_num).tolist()
            output["mean_ssim"] = (ssim / test_num).tolist()
            output["mean_lpips"] = (lpips / test_num).tolist()
            output["mean_depth_mse"] = (depth_mse / test_num).tolist()

        if self.eval_mode in ["complete", "mesh"]:
            print("evaluate mesh quality \n")
            for i, mesh in enumerate(self.mesh_list):
                mesh_rec = open3dmesh_2_trimesh(mesh)
                acc, comp, comp_ratio, chamfer_dist = calc_3d_mesh_metric(
                    mesh_rec, self.mesh_gt, dist_thres=0.02
                )
                accuracy[i] = acc
                completion[i] = comp
                completion_ratio[i] = comp_ratio
                chamfer_distance[i] = chamfer_dist

            output["mesh_accuracy"] = accuracy.tolist()
            output["mesh_completion"] = completion.tolist()
            output["mesh_completion_ratio"] = completion_ratio.tolist()
            output["mesh_chamfer_distance"] = chamfer_distance.tolist()

        print(output)
        return output
