# ----------------------------------------------#
# Pro    : cbct
# File   : utils_3d.py
# Date   : 2025
# Author : Adapted for 3D Cone-Beam CT
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


def cone_beam_ray(detector_u_pos, detector_v_pos, SOD, SDD):
    """
    Generate 3D rays for cone-beam CT geometry.

    Args:
        detector_u_pos: 1D array of detector positions along u (horizontal) axis in degrees
        detector_v_pos: 1D array of detector positions along v (vertical) axis in degrees
        SOD: Source-to-Origin Distance (isocenter distance)
        SDD: Source-to-Detector Distance

    Returns:
        rays: (num_det_v, num_det_u, num_samples, 3) array of 3D ray coordinates

    Note:
        - Source is at (0, -SOD, 0) in object space
        - Detector is a flat panel at distance SDD from source
        - Rays are sampled from source to detector, passing through reconstruction volume
    """
    # Source position at (0, -SOD, 0) - positioned along negative y-axis
    source_x = 0
    source_y = -1  # Normalized coordinate
    source_z = 0

    num_det_u = len(detector_u_pos)
    num_det_v = len(detector_v_pos)

    # Number of samples along each ray (similar to 2*SOD in 2D version)
    num_samples = int(2 * SOD)

    # Initialize ray array: (num_det_v, num_det_u, num_samples, 3)
    rays = np.zeros((num_det_v, num_det_u, num_samples, 3))

    # For each detector element, compute the ray from source to detector
    for iv in range(num_det_v):
        for iu in range(num_det_u):
            # Convert detector angles to positions on flat panel detector
            # Flat panel detector positioned at y = SOD * (SDD/SOD - 1)
            cone_angle_u = np.deg2rad(detector_u_pos[iu])
            cone_angle_v = np.deg2rad(detector_v_pos[iv])

            # Detector element position (flat panel geometry)
            # Using small angle approximation for typical CT geometry
            det_distance_normalized = SDD / SOD  # Normalized detector distance
            det_x = det_distance_normalized * np.tan(cone_angle_u)
            det_y = det_distance_normalized - 1  # Distance from source (normalized)
            det_z = det_distance_normalized * np.tan(cone_angle_v)

            # Create ray samples from source to beyond detector
            # Sample along the ray direction
            t = np.linspace(0, 2, num_samples)  # Parameter along ray (0 at source, 1 at detector)

            # Ray equation: P(t) = source + t * (detector - source)
            ray_x = source_x + t * (det_x - source_x)
            ray_y = source_y + t * (det_y - source_y)
            ray_z = source_z + t * (det_z - source_z)

            # Stack coordinates: (num_samples, 3)
            rays[iv, iu, :, 0] = ray_x
            rays[iv, iu, :, 1] = ray_y
            rays[iv, iu, :, 2] = ray_z

    return rays


def grid_coordinate_3d(h, w, d):
    """
    Generate 3D grid coordinates for reconstruction volume.

    Args:
        h: Height (number of voxels in x direction)
        w: Width (number of voxels in y direction)
        d: Depth (number of voxels in z direction)

    Returns:
        xyz: (h*w*d, 3) array of normalized 3D coordinates in [-1, 1]
    """
    x = np.linspace(-1, 1, h)
    y = np.linspace(-1, 1, w)
    z = np.linspace(-1, 1, d)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')  # (h, w, d) each
    xyz = np.stack([x, y, z], -1).reshape(-1, 3)  # (h*w*d, 3)
    return xyz


def rotate_ray_3d(xyz, angle, axis='z'):
    """
    Rotate 3D rays around specified axis for gantry rotation.

    Args:
        xyz: (..., 3) array of 3D coordinates
        angle: Rotation angle in degrees
        axis: Rotation axis ('x', 'y', or 'z'). Default 'z' for typical CT gantry rotation

    Returns:
        xyz_rotated: Rotated coordinates with same shape as input
    """
    xyz_shape = xyz.shape
    angle_rad = np.deg2rad(angle)

    # Rotation matrices for each axis
    if axis == 'z':
        # Rotation around z-axis (typical gantry rotation)
        trans_mat = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad),  np.cos(angle_rad), 0],
            [0,                  0,                 1]
        ])
    elif axis == 'y':
        # Rotation around y-axis
        trans_mat = np.array([
            [ np.cos(angle_rad), 0, np.sin(angle_rad)],
            [ 0,                 1, 0                ],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])
    elif axis == 'x':
        # Rotation around x-axis
        trans_mat = np.array([
            [1, 0,                  0                ],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad),  np.cos(angle_rad)]
        ])
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'")

    # Reshape to (N, 3), apply rotation, reshape back
    xyz = xyz.reshape(-1, 3)
    xyz_rotated = np.dot(xyz, trans_mat.T).reshape(xyz_shape)
    return xyz_rotated


# Keep 2D functions for backward compatibility
def fan_beam_ray(proj_pos, SOD):
    """Original 2D fan-beam ray generation (preserved for compatibility)"""
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
    """Original 2D grid generation (preserved for compatibility)"""
    x = np.linspace(-1, 1, h)
    y = np.linspace(-1, 1, w)
    x, y = np.meshgrid(x, y, indexing='ij')  # (h, w), (h, w)
    xy = np.stack([x, y], -1).reshape(-1, 2)  # (h*w, 2)
    return xy


def rotate_ray(xy, angle):
    """Original 2D rotation (preserved for compatibility)"""
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
