"""
Diagnostic: pinpoints which stage in reprojection produces all zeros.
Run from the repository root with the trained model in model/model_0.pkl.
"""
import os
import torch
import numpy as np
import SimpleITK as sitk
import tinycudann as tcnn
import commentjson as json

import utils

with open("config.json") as f:
    cfg = json.load(f)

in_path    = cfg["file"]["in_dir"]
SOD        = cfg["file"]["SOD"]
voxel_size = cfg["file"]["voxel_size"]
e_level    = 6
energy_idx = 2
device     = torch.device("cuda:0")

# 1. Load network
net = tcnn.NetworkWithInputEncoding(3, e_level, cfg["encoding"], cfg["network"]).to(device)
state = torch.load("./model/model_0.pkl", map_location=device)
print("[1] state_dict keys:", list(state.keys()))
for k, v in state.items():
    print(f"    {k}: shape={tuple(v.shape)}, dtype={v.dtype}, "
          f"min={v.float().min().item():.4g}, max={v.float().max().item():.4g}, "
          f"abs_mean={v.float().abs().mean().item():.4g}")
net.load_state_dict(state)
net.eval()

# 2. Probe network at a few points inside the volume
with torch.no_grad():
    for label, pts in [
        ("origin (0,0,0)",      torch.zeros(1024, 3, device=device)),
        ("near origin (±0.05)", (torch.rand(1024, 3, device=device) - 0.5) * 0.1),
        ("full [-1,1]",         (torch.rand(8192, 3, device=device) - 0.5) * 2),
    ]:
        out = net(pts).float()
        print(f"[2] {label}: out shape={tuple(out.shape)} "
              f"min={out.min().item():.4g} max={out.max().item():.4g} "
              f"mean={out.mean().item():.4g}  "
              f"per-energy mean={out.mean(0).cpu().numpy()}")

# 3. Run a single angle of reprojection (no mask) and inspect mu before sum
proj_pos_u = sitk.GetArrayFromImage(sitk.ReadImage(
    f'{in_path}/fanSensorPosition_fanangle_32f.nii')).reshape(-1)
proj_pos_v = sitk.GetArrayFromImage(sitk.ReadImage(
    f'{in_path}/fanSensorPosition_coneangle_32f.nii')).reshape(-1)
rays = utils.cone_beam_ray(proj_pos_u, proj_pos_v, SOD)
rays = rays.transpose(1, 0, 2, 3).reshape(-1, rays.shape[2], 3)   # (n_det, n_samples, 3)
rays_rot = utils.rotate_ray_3d(rays.copy(), angle=0.0)
ray_flat = rays_rot.reshape(-1, 3)
print(f"[3] ray_flat: shape={ray_flat.shape}, "
      f"x∈[{ray_flat[:,0].min():.3f},{ray_flat[:,0].max():.3f}], "
      f"y∈[{ray_flat[:,1].min():.3f},{ray_flat[:,1].max():.3f}], "
      f"z∈[{ray_flat[:,2].min():.3f},{ray_flat[:,2].max():.3f}]")

with torch.no_grad():
    pts = torch.from_numpy(ray_flat[:200000]).float().to(device)
    mu_full = net(pts).float()
    mu = mu_full[:, energy_idx]
    print(f"[3] mu_full[:,{energy_idx}]: min={mu.min().item():.4g}, "
          f"max={mu.max().item():.4g}, mean={mu.mean().item():.4g}, "
          f"nonzero={(mu != 0).sum().item()}/{mu.numel()}")
    print(f"[3] mu_full all energies mean: {mu_full.mean(0).cpu().numpy()}")
