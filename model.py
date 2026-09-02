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
        num_points = ray.shape[0]  # total number of sample points: batch_size * num_sample_ray * k
        
        # F.grid_sample 5D convention:
        #   - input: (N, C, D, H, W)
        #   - grid:  (N, D_out, H_out, W_out, 3), where (x, y, z) index (W, H, D)
        #
        # self.mask is created as (1, 1, h, w, d) where (h, w, d) = (row, col, slice)
        # To match PyTorch's (D, H, W) = (depth, height, width) convention:
        #   - D should correspond to slice (d) -> z coordinate
        #   - H should correspond to row (h)   -> y coordinate
        #   - W should correspond to col (w)   -> x coordinate
        # Permute mask from (N, C, h, w, d) to (N, C, d, h, w)
        mask_permuted = self.mask.permute(0, 1, 4, 2, 3)  # (1, 1, d, h, w)
        
        # Reshape ray to 5D grid: (N, D_out, H_out, W_out, 3)
        # Assuming ray coordinates are (x, y, z) = (col, row, slice) in normalized [-1, 1]
        grid = ray.view(1, num_points, 1, 1, 3)
        
        # Sample mask values at ray positions
        # Output shape: (1, 1, num_points, 1, 1)
        mask_sampled = F.grid_sample(
            mask_permuted, grid, mode='nearest', align_corners=False
        )
        
        # Reshape mask back to (batch_size, num_sample_ray, k)
        mask_sampled = mask_sampled.view(batch_size, num_sample_ray, k)
        
        # Compute attenuation smoothness loss over energy levels
        diff = torch.sum(torch.abs(intensity[:, :, :, 1:] - intensity[:, :, :, :e_level-1]), dim=-1) * mask_sampled
        return self.lamb * torch.sum(diff) / (batch_size * num_sample_ray * k)