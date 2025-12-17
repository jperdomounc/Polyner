# ----------------------------------------------#
# Pro    : cbct
# File   : dataset_3d.py
# Date   : 2025
# Author : Adapted for 3D Cone-Beam CT
# ----------------------------------------------#
import utils_3d
import numpy as np
import SimpleITK as sitk
from torch.utils import data


class TrainData3D(data.Dataset):
    """
    3D Cone-Beam CT training dataset.

    Input data format (as stored in file):
        - projections: 3D array (num_det_v, num_angles, num_det_u)
        - detector positions: Two 1D arrays for u and v detector coordinates

    Internal format after transpose:
        - projections: 3D array (num_angles, num_det_v, num_det_u)
    """
    def __init__(self, proj_path, proj_u_pos_path, proj_v_pos_path,
                 num_sample_ray, num_angle, SOD, SDD, voxel_size, num_samples=200,
                 vol_dims=None):
        self.num_angle = num_angle
        self.num_sample_ray = num_sample_ray
        self.SOD = SOD
        self.SDD = SDD
        self.voxel_size = voxel_size
        self.num_samples = num_samples
        self.vol_dims = vol_dims  # (h, w, d) for proper z-scaling
        self.angles = np.linspace(0., 360., num=self.num_angle, endpoint=False)  # (num_angle, )

        # Load detector positions
        # Handle both 1D and meshgrid (2D) formats
        proj_u_data = sitk.GetArrayFromImage(sitk.ReadImage(proj_u_pos_path))
        proj_v_data = sitk.GetArrayFromImage(sitk.ReadImage(proj_v_pos_path))

        # If meshgrid format (2D), extract 1D arrays
        if len(proj_u_data.shape) == 2:
            # Fan angles (U): vary along first dimension (rows), extract first column
            self.proj_u_pos = proj_u_data[:, 0]  # (num_det_u, )
        else:
            self.proj_u_pos = proj_u_data.reshape(-1)

        if len(proj_v_data.shape) == 2:
            # Cone angles (V): vary along second dimension (columns), extract first row
            self.proj_v_pos = proj_v_data[0, :]  # (num_det_v, )
        else:
            self.proj_v_pos = proj_v_data.reshape(-1)

        self.num_det_u = len(self.proj_u_pos)
        self.num_det_v = len(self.proj_v_pos)

        # Load projections
        proj_data = sitk.GetArrayFromImage(sitk.ReadImage(proj_path))
        if len(proj_data.shape) != 3:
            raise ValueError(f"Expected 3D projection data, got shape {proj_data.shape}")

        # Data comes in as (num_det_v, num_angles, num_det_u) = (126, 360, 334)
        # We need (num_angles, num_det_v, num_det_u) for proper indexing
        # Check shape and transpose if needed
        if proj_data.shape[0] == self.num_det_v and proj_data.shape[1] == num_angle:
            # Data is (num_det_v, num_angles, num_det_u) - transpose to (num_angles, num_det_v, num_det_u)
            self.proj = np.transpose(proj_data, (1, 0, 2))
            print(f"Transposed projections: {proj_data.shape} -> {self.proj.shape}")
        elif proj_data.shape[0] == num_angle:
            # Data is already (num_angles, num_det_v, num_det_u)
            self.proj = proj_data
        else:
            raise ValueError(f"Unexpected projection shape {proj_data.shape}. "
                           f"Expected first dim to be num_det_v={self.num_det_v} or num_angle={num_angle}")

        # Generate 3D cone-beam rays
        # Shape: (num_det_v, num_det_u, num_samples, 3)
        self.rays = utils_3d.cone_beam_ray(self.proj_u_pos, self.proj_v_pos, self.SOD, self.SDD,
                                            self.num_samples, vol_dims=self.vol_dims)

        # For random sampling, we'll sample from the u direction
        self.index_max_u = self.num_det_u - self.num_sample_ray
        # We could also sample in v direction, but for simplicity start with u

    def __getitem__(self, item):
        ang = self.angles[item]
        proj = self.proj[item]  # (num_det_v, num_det_u)

        # Sample rays along u direction (horizontal)
        # Randomly select a v-row and consecutive u-columns
        index_v = np.random.randint(0, self.num_det_v, size=1)[0]
        index_u = np.random.randint(0, self.index_max_u, size=1)[0] # potentially wrong - juan

        # Sample consecutive rays: (num_sample_ray, num_samples, 3)
        ray_sample = self.rays[index_v, index_u:index_u+self.num_sample_ray]

        # Sample corresponding projections: (num_sample_ray, )
        proj_sample = proj[index_v, index_u:index_u+self.num_sample_ray]

        # Rotate rays for current gantry angle
        ray_sample = utils_3d.rotate_ray_3d(xyz=ray_sample, angle=ang, axis='z')

        return ray_sample, proj_sample

    def __len__(self):
        return self.num_angle


class TestData3D(data.Dataset):
    """
    3D reconstruction volume dataset for inference.
    """
    def __init__(self, h, w, d):
        self.h, self.w, self.d = h, w, d
        # Generate 3D grid: (h*w*d, 3)
        self.xyz = utils_3d.grid_coordinate_3d(h=self.h, w=self.w, d=self.d).reshape(1, int(h*w*d), 3)

    def __getitem__(self, item):
        return self.xyz[item]  # (h*w*d, 3)

    def __len__(self):
        return 1


# 2D versions preserved for backward compatibility
class TrainData(data.Dataset):
    """Original 2D fan-beam training dataset"""
    def __init__(self, proj_path, proj_pos_path, num_sample_ray, num_angle, SOD, voxel_size):
        self.num_angle = num_angle
        self.num_sample_ray = num_sample_ray
        self.SOD = SOD
        self.voxel_size = voxel_size
        self.angles = np.linspace(0., 360., num=self.num_angle, endpoint=False)  # (num_angle, )
        self.proj_pos = sitk.GetArrayFromImage(sitk.ReadImage(proj_pos_path)).reshape(-1) # (num_det, )
        self.num_det = len(self.proj_pos)
        # projection, i.e., sinogram & metal_trace
        self.proj = sitk.GetArrayFromImage(sitk.ReadImage(proj_path))  # (num_angle, num_det)
        # ray
        self.rays = utils_3d.fan_beam_ray(self.proj_pos, self.SOD) # (num_det, 2*SOD, 2)
        self.index_max = self.num_det - self.num_sample_ray

    def __getitem__(self, item):
        ang = self.angles[item]
        proj = self.proj[item].reshape(-1, )  # (num_det, )
        # sample ray, projection, and metal trace
        index = np.random.randint(0, self.index_max, size=1)[0]
        ray_sample = self.rays[index:index+self.num_sample_ray]     # (num_sample_ray, 2*SOD, 2)
        proj_sample = proj[index:index+self.num_sample_ray]     # (num_sample_ray, )
        # rotate ray
        ray_sample = utils_3d.rotate_ray(xy=ray_sample, angle=ang)
        return ray_sample, proj_sample

    def __len__(self):
        return self.num_angle


class TestData(data.Dataset):
    """Original 2D test dataset"""
    def __init__(self, h, w):
        self.h, self.w = h, w
        self.xy = utils_3d.grid_coordinate(h=self.h, w=self.w).reshape(1, int(h*w), 2)

    def __getitem__(self, item):
        return self.xy[item]    # (h*w, 2)

    def __len__(self):
        return 1
