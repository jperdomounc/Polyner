# ----------------------------------------------#
# Pro    : cbct
# File   : dataset.py
# Date   : 2023/2/22
# Author : Qing Wu
# Email  : wuqing@shanghaitech.edu.cn
# ----------------------------------------------#
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio


def psnr(image, ground_truth):
    data_range = np.max(ground_truth) - np.min(ground_truth)
    return peak_signal_noise_ratio(ground_truth, image, data_range=data_range)


def ssim(image, ground_truth):
    data_range = np.max(ground_truth) - np.min(ground_truth)
    return structural_similarity(image, ground_truth, data_range=data_range)


def fan_beam_ray(proj_pos, SOD):
    origin_x = 0
    origin_y = -1
    y = np.linspace(-1, 1, int(2*SOD)).reshape(-1, 1)  # (2*SOD, ) -> (2*SOD, 1)
    x = np.zeros_like(y)  # (2*SOD, 1)
    xy_temp = np.concatenate((x, y), axis=-1)  # (2*SOD, 2)
    xy_temp = np.concatenate((xy_temp, np.ones_like(x)), axis=-1)  # (2*SOD, 3)
    num_det = len(proj_pos)
    xy = np.zeros(shape=(num_det, int(2*SOD), 2)) # (L, 2*SOD, 2)
    for i in range(num_det):
        fan_angle_rad = np.deg2rad(proj_pos[num_det-i-1])
        M = np.array(
            [
                [np.cos(fan_angle_rad), -np.sin(fan_angle_rad),
                 -1*origin_x*np.cos(fan_angle_rad)+origin_y*np.sin(fan_angle_rad)+origin_x],
                [np.sin(fan_angle_rad), np.cos(fan_angle_rad),
                 -1*origin_x*np.sin(fan_angle_rad)-origin_y*np.cos(fan_angle_rad)+origin_y],
                [0, 0, 1]
            ]
        )
        temp = xy_temp @ M.T # (2*SOD, 3) @ (3, 3) -> (2*SOD, 3)
        xy[i, :, :] = temp[:, :2] # (2*SOD, 2)
    return xy


def grid_coordinate(h, w):
    x = np.linspace(-1, 1, h)
    y = np.linspace(-1, 1, w)
    x, y = np.meshgrid(x, y, indexing='ij')  # (h, w), (h, w)
    xy = np.stack([x, y], -1).reshape(-1, 2)  # (h*w, 2)
    return xy


def rotate_ray(xy, angle):
    xy_shape = xy.shape
    angle_rad = np.deg2rad(angle)
    trans_mat = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad),  np.cos(angle_rad)],
        ]
    )
    xy = xy.reshape(-1, 2)
    xy = (np.dot(xy, trans_mat.T)).reshape(xy_shape)
    return xy


def cone_beam_ray(detector_h, detector_w, SOD, SDD, cone_angle):
    """Generate 3D cone beam rays from source to detector pixels"""
    # Source position
    source_pos = np.array([0, -SOD, 0])  # (x, y, z)

    # Detector plane positions
    detector_distance = SDD - SOD
    det_y = detector_distance

    # Create detector grid
    cone_angle_rad = np.deg2rad(cone_angle)
    det_size_x = 2 * detector_distance * np.tan(cone_angle_rad / 2)
    det_size_z = det_size_x  # Square detector

    x_det = np.linspace(-det_size_x/2, det_size_x/2, detector_w)
    z_det = np.linspace(-det_size_z/2, det_size_z/2, detector_h)
    x_det_grid, z_det_grid = np.meshgrid(x_det, z_det, indexing='ij')

    # Detector pixel positions
    detector_positions = np.stack([
        x_det_grid.flatten(),
        np.full(detector_h * detector_w, det_y),
        z_det_grid.flatten()
    ], axis=1)  # (detector_h * detector_w, 3)

    # Generate rays from source to each detector pixel
    num_samples_per_ray = int(2 * SOD)  # Similar to 2D case
    rays = np.zeros((detector_h * detector_w, num_samples_per_ray, 3))

    for i, det_pos in enumerate(detector_positions):
        # Ray direction from source to detector pixel
        ray_dir = det_pos - source_pos
        ray_dir = ray_dir / np.linalg.norm(ray_dir)

        # Sample points along the ray
        t = np.linspace(-SOD, SOD, num_samples_per_ray)
        ray_points = source_pos[np.newaxis, :] + t[:, np.newaxis] * ray_dir[np.newaxis, :]

        # Normalize coordinates to [-1, 1] range
        ray_points[:, 0] /= SOD  # x
        ray_points[:, 1] /= SOD  # y
        ray_points[:, 2] /= SOD  # z

        rays[i] = ray_points

    return rays


def grid_coordinate_3d(h, w, d):
    """Generate 3D grid coordinates"""
    x = np.linspace(-1, 1, h)
    y = np.linspace(-1, 1, w)
    z = np.linspace(-1, 1, d)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')  # (h, w, d), (h, w, d), (h, w, d)
    xyz = np.stack([x, y, z], -1).reshape(-1, 3)  # (h*w*d, 3)
    return xyz


def rotate_ray_3d(xyz, angle, axis='z'):
    """Rotate 3D rays around specified axis"""
    xyz_shape = xyz.shape
    angle_rad = np.deg2rad(angle)

    if axis == 'z':
        # Rotation around z-axis (typical for CT gantry rotation)
        trans_mat = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad),  np.cos(angle_rad), 0],
            [0,                  0,                 1]
        ])
    elif axis == 'x':
        trans_mat = np.array([
            [1, 0,                  0                ],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad),  np.cos(angle_rad)]
        ])
    elif axis == 'y':
        trans_mat = np.array([
            [ np.cos(angle_rad), 0, np.sin(angle_rad)],
            [ 0,                 1, 0                ],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])

    xyz = xyz.reshape(-1, 3)
    xyz = (np.dot(xyz, trans_mat.T)).reshape(xyz_shape)
    return xyz