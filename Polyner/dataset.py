# ----------------------------------------------#
# Pro    : cbct
# File   : dataset.py
# Date   : 2023/2/22
# Author : Qing Wu
# Email  : wuqing@shanghaitech.edu.cn
# ----------------------------------------------#
import utils
import numpy as np
import SimpleITK as sitk
from torch.utils import data

class TrainData(data.Dataset):
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
        self.rays = utils.fan_beam_ray(self.proj_pos, self.SOD) # (num_det, 2*SOD, 2)
        self.index_max = self.num_det - self.num_sample_ray

    def __getitem__(self, item):
        ang = self.angles[item]
        proj = self.proj[item].reshape(-1, )  # (num_det, )
        # sample ray, projection, and metal trace
        index = np.random.randint(0, self.index_max, size=1)[0]
        ray_sample = self.rays[index:index+self.num_sample_ray]     # (num_sample_ray, 2*SOD, 2)
        proj_sample = proj[index:index+self.num_sample_ray]     # (num_sample_ray, )
        # rotate ray
        ray_sample = utils.rotate_ray(xy=ray_sample, angle=ang)
        return ray_sample, proj_sample

    def __len__(self):
        return self.num_angle


class TestData(data.Dataset):
    def __init__(self, h, w):
        self.h, self.w = h, w
        self.xy = utils.grid_coordinate(h=self.h, w=self.w).reshape(1, int(h*w), 2)

    def __getitem__(self, item):
        return self.xy[item]    # (h*w, 2)

    def __len__(self):
        return 1


class TrainData3D(data.Dataset):
    def __init__(self, proj_path, proj_pos_path, num_sample_ray, num_angle, SOD, SDD,
                 detector_h, detector_w, cone_angle, voxel_size):
        self.num_angle = num_angle
        self.num_sample_ray = num_sample_ray
        self.SOD = SOD
        self.SDD = SDD
        self.detector_h = detector_h
        self.detector_w = detector_w
        self.cone_angle = cone_angle
        self.voxel_size = voxel_size

        # Projection angles for gantry rotation
        self.angles = np.linspace(0., 360., num=self.num_angle, endpoint=False)

        # For 3D cone beam, we don't use proj_pos but generate cone beam geometry
        # Load projection data (now 3D: num_angle x detector_h x detector_w)
        proj_data = sitk.GetArrayFromImage(sitk.ReadImage(proj_path))
        if len(proj_data.shape) == 2:  # If 2D, extend to 3D
            self.proj = np.repeat(proj_data[:, :, np.newaxis], detector_h, axis=2)
        else:
            self.proj = proj_data

        # Generate 3D cone beam rays
        self.rays = utils.cone_beam_ray(detector_h, detector_w, SOD, SDD, cone_angle)
        self.num_det_pixels = detector_h * detector_w
        self.index_max = self.num_det_pixels - self.num_sample_ray

    def __getitem__(self, item):
        ang = self.angles[item]
        # For 3D, proj shape is (detector_h, detector_w)
        proj = self.proj[item].flatten()  # Flatten to 1D

        # Sample rays and corresponding projections
        index = np.random.randint(0, self.index_max, size=1)[0]
        ray_sample = self.rays[index:index+self.num_sample_ray]  # (num_sample_ray, ray_length, 3)
        proj_sample = proj[index:index+self.num_sample_ray]  # (num_sample_ray,)

        # Rotate rays around z-axis for gantry rotation
        ray_sample = utils.rotate_ray_3d(xyz=ray_sample, angle=ang, axis='z')

        return ray_sample, proj_sample

    def __len__(self):
        return self.num_angle


class TestData3D(data.Dataset):
    def __init__(self, h, w, d):
        self.h, self.w, self.d = h, w, d
        self.xyz = utils.grid_coordinate_3d(h=self.h, w=self.w, d=self.d).reshape(1, int(h*w*d), 3)

    def __getitem__(self, item):
        return self.xyz[item]  # (h*w*d, 3)

    def __len__(self):
        return 1