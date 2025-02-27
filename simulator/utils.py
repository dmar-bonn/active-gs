import numpy as np
import habitat_sim
import torch


SENSOR_TYPE = {
    "color": habitat_sim.SensorType.COLOR,
    "depth": habitat_sim.SensorType.DEPTH,
    "semantic": habitat_sim.SensorType.SEMANTIC,
}


def compute_camera_intrinsic(h, w, vfov, hfov, normalize=True):
    vfov_rad = np.radians(vfov)
    hfov_rad = np.radians(hfov)

    fx = (w / 2) / np.tan(hfov_rad / 2)
    fy = (h / 2) / np.tan(vfov_rad / 2)

    cx = w / 2
    cy = h / 2

    if normalize:
        fx /= w
        cx /= w
        fy /= h
        cy /= h

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).astype(np.float32)
    return torch.from_numpy(K)


def opencv_to_opengl_camera(transform=None):
    if transform is None:
        transform = np.eye(4)
    return transform @ np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def opengl_to_opencv_camera(transform=None):
    if transform is None:
        transform = np.eye(4)
    return transform @ np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
