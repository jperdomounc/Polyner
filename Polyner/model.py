# ----------------------------------------------#
# Pro    : cbct
# File   : dataset.py
# Date   : 2023/2/22
# Author : Qing Wu
# Email  : wuqing@shanghaitech.edu.cn
# ----------------------------------------------#
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attenuation_Smootion_Over_Energies_Loss(nn.Module):
    def __init__(self, mask, lamb):
        super(Attenuation_Smootion_Over_Energies_Loss, self).__init__()
        self.mask = mask
        self.lamb = lamb
    def forward(self, ray, intensity):
        batch_size, num_sample_ray, k, e_level = intensity.shape
        # For 3D cone beam, ray coordinates are (x, y, z)
        # Normalize ray coordinates to [-1, 1] range for grid_sample
        ray_normalized = ray.clone()
        ray_normalized = ray_normalized / (ray_normalized.abs().max() + 1e-8) * 0.9

        # Handle 3D mask sampling
        if len(self.mask.shape) == 5:  # 3D mask: (1, 1, D, H, W)
            mask = F.grid_sample(
                self.mask, ray_normalized.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                mode='nearest', align_corners=False
            )[0, 0, 0, 0, :].view(batch_size, num_sample_ray, k)
        else:  # 2D mask fallback
            mask = F.grid_sample(
                self.mask, ray_normalized[:, :2].unsqueeze(0).unsqueeze(0),
                mode='nearest', align_corners=False
            )[0, 0, 0, :].view(batch_size, num_sample_ray, k)

        diff = torch.sum(torch.abs(intensity[:, :, :, 1:] - intensity[:, :, :, :e_level-1]), dim=-1) * mask
        return self.lamb * torch.sum(diff) / (batch_size * num_sample_ray * k)
