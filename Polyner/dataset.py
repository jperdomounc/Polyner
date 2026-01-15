import utils
import numpy as np
import SimpleITK as sitk
from torch.utils import data


class TrainData(data.Dataset):
    def __init__(self, proj_path, proj_pos_path_u, proj_pos_path_v, num_sample_ray, num_angle, SOD, voxel_size):
        self.num_angle = num_angle
        self.num_sample_ray = num_sample_ray
        self.SOD = SOD
        self.voxel_size = voxel_size
        self.angles = np.linspace(0., 360., num=self.num_angle, endpoint=False)  # (num_angle,)
        
        # Detector positions
        self.proj_pos_u = sitk.GetArrayFromImage(sitk.ReadImage(proj_pos_path_u)).reshape(-1)  # (num_det_col,)
        self.proj_pos_v = sitk.GetArrayFromImage(sitk.ReadImage(proj_pos_path_v)).reshape(-1)  # (num_det_row,)
        self.num_det_u = len(self.proj_pos_u)  # number of detector columns
        self.num_det_v = len(self.proj_pos_v)  # number of detector rows
        self.num_det = self.num_det_u * self.num_det_v  # total detector pixels
        
        # Load CBCT projection data: (num_det_row, num_det_col, num_angle)
        self.proj = sitk.GetArrayFromImage(sitk.ReadImage(proj_path))
        print(f'Original shape: {self.proj.shape}')  # (num_angle, num_det_col, num_det_row)
        self.proj = self.proj.transpose(2, 1, 0)  # (num_det_row, num_det_col, num_angle)
        print(f'Transposed shape: {self.proj.shape}')  # (num_det_row, num_det_col, num_angle)

        # Generate cone beam rays: (num_det_u, num_det_v, 2*SOD, 3) = (num_det_col, num_det_row, num_samples, 3)
        rays = utils.cone_beam_ray(self.proj_pos_u, self.proj_pos_v, self.SOD)
        
        # Transpose rays from (num_det_u, num_det_v, ...) to (num_det_v, num_det_u, ...)
        # This aligns with projection data ordering: (num_det_row, num_det_col, ...)
        rays = rays.transpose(1, 0, 2, 3)  # (num_det_row, num_det_col, num_samples, 3)
        
        # Flatten to (num_det_row * num_det_col, num_samples, 3) for easier sampling
        self.num_samples = rays.shape[2]
        self.rays = rays.reshape(-1, self.num_samples, 3)  # (num_det, num_samples, 3)
        
        self.index_max = self.num_det - self.num_sample_ray

    def __getitem__(self, item):
        ang = self.angles[item]
        
        # Get projection for this angle: index the last dimension (angle)
        # proj shape: (num_det_row, num_det_col, num_angle) -> select angle -> (num_det_row, num_det_col)
        proj = self.proj[:, :, item].reshape(-1)  # (num_det_row * num_det_col,) = (num_det,)
        
        # Randomly sample contiguous detector rays
        index = np.random.randint(0, self.index_max)
        ray_sample = self.rays[index:index + self.num_sample_ray]  # (num_sample_ray, num_samples, 3)
        proj_sample = proj[index:index + self.num_sample_ray]  # (num_sample_ray,)
        
        # Rotate ray coordinates for this gantry angle
        ray_sample = utils.rotate_ray_3d(xyz=ray_sample, angle=ang)
        
        return ray_sample, proj_sample

    def __len__(self):
        return self.num_angle


class TestData(data.Dataset):
    def __init__(self, h, w, d):
        """
        Generate 3D grid coordinates for volume reconstruction
        
        Parameters:
        -----------
        h, w, d : int
            Volume dimensions (height, width, depth)
        """
        self.h, self.w, self.d = h, w, d
        # Generate 3D grid coordinates: (h*w*d, 3)
        self.xyz = utils.grid_coordinate_3d(h=self.h, w=self.w, d=self.d)
        # Reshape to (1, h*w*d, 3) for batch compatibility
        self.xyz = self.xyz.reshape(1, h * w * d, 3)

    def __getitem__(self, item):
        return self.xyz[item]  # (h*w*d, 3)

    def __len__(self):
        return 1
