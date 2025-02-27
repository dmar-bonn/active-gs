import os
import hydra
import numpy as np
import torch
import pickle
from utils.operations import GaussianRenderer
import open3d as o3d

from mapping.gaussian_map import GaussianMap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import warnings

warnings.simplefilter("ignore")


H, W = 1024, 1024  # resolution for tsdf fusion


@hydra.main(
    version_base=None,
    config_path="./config",
    config_name="main",
)
def main(cfg):
    experiment_path = os.path.join(
        cfg.experiment.output_dir,
        str(cfg.experiment.exp_id),
        cfg.scene.scene_name,
        cfg.planner.planner_name,
        str(cfg.experiment.run_id),
    )

    record_info_file = f"{experiment_path}/map/record_info.txt"
    if os.path.exists(record_info_file):
        record_info = np.loadtxt(record_info_file)
    else:
        print("no record file!!!")
        return False

    id_list = record_info[:, 0]
    for map_id in id_list:
        print(f"generatinig mesh for gaussian map {map_id}")
        map_id = int(map_id)
        map_file = f"{experiment_path}/map/map_{map_id:03}.th"
        os.path.exists(map_file)

        gaussain_map = GaussianMap(None, device)
        gaussain_map.load(map_file)

        camera_file = f"{experiment_path}/map/cameras_{map_id:03}.pkl"
        os.path.exists(camera_file)
        with open(camera_file, "rb") as pickle_file:
            camera_params = pickle.load(pickle_file)

        mesh_file = f"{experiment_path}/map/mesh_{map_id:03}.ply"
        mesh = generate_mesh(gaussain_map, camera_params)
        o3d.io.write_triangle_mesh(mesh_file, mesh)


def generate_mesh(gaussian_map, camera_params):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.02,
        sdf_trunc=0.1,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for camera_view in camera_params:
        extrinsic = torch.tensor(camera_view[0:16], device=device).view(4, 4)
        intrinsic = torch.tensor(camera_view[16:], device=device).view(3, 3)

        rgb, depth, _, _, _, _, _, _, _ = GaussianRenderer(
            extrinsic.unsqueeze(0),
            intrinsic.unsqueeze(0),
            gaussian_map.get_attr(),
            gaussian_map.background_color,
            (gaussian_map.scene_near, gaussian_map.scene_far),
            (H, W),
            device,
        ).render_view_all()

        rgb = rgb.squeeze(0).permute(1, 2, 0).cpu().numpy()
        depth = depth.squeeze(0).squeeze(0).cpu().numpy()
        intrinsic = intrinsic.cpu().numpy()
        extrinsic = extrinsic.cpu().numpy()

        intrinsic[0, :] *= W
        intrinsic[1, :] *= H
        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, intrinsic)
        rgb = np.ascontiguousarray(rgb * 255).astype(np.uint8)
        depth = np.ascontiguousarray(depth).astype(np.float32)
        rgb = o3d.geometry.Image(rgb)
        depth = o3d.geometry.Image(depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth, depth_scale=1, depth_trunc=10.0, convert_rgb_to_intensity=False
        )
        volume.integrate(rgbd, intrinsic, np.linalg.inv(extrinsic))
    mesh = volume.extract_triangle_mesh()
    mesh = filter_isolated_vertices(mesh)
    return mesh


def filter_isolated_vertices(mesh, filter_cluster_min_tri=50):
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    triangles_to_remove = (
        cluster_n_triangles[triangle_clusters] < filter_cluster_min_tri
    )
    mesh.remove_triangles_by_mask(triangles_to_remove)
    return mesh


if __name__ == "__main__":
    main()
