# ----------------------------------------------#
# Reprojection: use a trained PolyNER model to
# generate dense-view cone-beam projections.
# Optional: skip metal voxels (treat as air)
# by zeroing mu at sample points inside a 3D
# binary metal mask.
# ----------------------------------------------#
import os
import utils
import torch
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
import tinycudann as tcnn
import commentjson as json
from tqdm import tqdm


def load_metal_mask(mask_path, h, w, d, SOD, device, undo_y_flip=True):
    """
    Load a 3D binary metal mask and prepare it for grid_sample lookups
    in the network's normalized [-1, 1] coordinate frame.

    Returns (mask_tensor, lo, hi):
        mask_tensor : (1, 1, d, w, h) float tensor on `device`
                      arranged so the last 3 dims map to (z, y, x) for grid_sample
        lo, hi      : length-3 tensors giving the (x, y, z) extent of the mask
                      within the network's [-1, 1] frame, used to rescale query pts
    """
    mask_np = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).astype(np.float32)
    # SimpleITK returns arrays in (z, y, x) order. Our volume convention is
    # (h, w, d) = (x, y, z), so transpose to match.
    if mask_np.shape == (d, w, h):
        mask_np = mask_np.transpose(2, 1, 0).copy()
    assert mask_np.shape == (h, w, d), (
        f"Mask shape {mask_np.shape} != expected volume shape ({h},{w},{d}). "
        f"Mask must be in (h, w, d) or (d, w, h) ordering."
    )

    # test.py applies np.flip(img_pre, axis=1) before saving the reconstruction.
    # If the mask was drawn to align with that saved volume, we must undo the flip
    # to get back into the network's coordinate frame. Toggle with `undo_y_flip`.
    if undo_y_flip:
        mask_np = np.flip(mask_np, axis=1).copy()

    # grid_sample expects (N, C, D, H, W) with the query grid's last dim = (x, y, z).
    # Our mask is (h=x, w=y, d=z); permute so axes become (d=z, w=y, h=x) = (D, H, W).
    mask_tensor = torch.from_numpy(mask_np).permute(2, 1, 0).contiguous()
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, d, w, h)

    # The network trained on coords from np.linspace(-1, 1, 2*SOD+1).
    # The reconstructed (h, w, d) volume is cropped at kx, ky, kz.
    # linspace(-1, 1, 2*SOD+1)[i] = -1 + i/SOD, so the mask spans:
    kx = int(1 + ((2 * SOD) - h) / 2)
    ky = int(((2 * SOD) - w) / 2)
    kz = int(((2 * SOD) - d) / 2)
    x_lo, x_hi = -1.0 + kx / SOD, -1.0 + (kx + h - 1) / SOD
    y_lo, y_hi = -1.0 + ky / SOD, -1.0 + (ky + w - 1) / SOD
    z_lo, z_hi = -1.0 + kz / SOD, -1.0 + (kz + d - 1) / SOD
    lo = torch.tensor([x_lo, y_lo, z_lo], device=device, dtype=torch.float32)
    hi = torch.tensor([x_hi, y_hi, z_hi], device=device, dtype=torch.float32)

    print(f"Metal mask loaded: shape {mask_np.shape}, "
          f"num metal voxels = {int(mask_np.sum())}, "
          f"extent in [-1,1] frame: x={x_lo:.3f}..{x_hi:.3f}, "
          f"y={y_lo:.3f}..{y_hi:.3f}, z={z_lo:.3f}..{z_hi:.3f}")

    return mask_tensor, lo, hi


def points_in_metal(pts, mask_tensor, lo, hi):
    """
    Given ray sample points in the network's [-1, 1] frame, return a bool tensor
    marking which points fall inside the metal mask.
    """
    pts_rescaled = (pts - lo) / (hi - lo) * 2.0 - 1.0  # (N, 3), each in [-1, 1] over the mask
    grid = pts_rescaled.view(1, -1, 1, 1, 3)           # (1, N, 1, 1, 3)
    sampled = F.grid_sample(
        mask_tensor, grid,
        mode='nearest', padding_mode='zeros', align_corners=True
    )
    return sampled.view(-1) > 0.5


def reproject(config, reproject_config):
    """
    Load a trained PolyNER model and generate dense-view projections
    via forward ray integration. Optionally skip metal voxels using a mask.

    Extra reproject_config keys for metal masking:
    ---------------------------------------------
    metal_mask_path : str or None
        Path to a 3D binary metal mask (.nii) in (h, w, d) space.
        If None, no masking is applied.
    mask_undo_y_flip : bool (default True)
        Whether to undo test.py's axis-1 flip when loading the mask.
    """

    # =============================================
    # 1. Read geometry parameters from config
    # =============================================
    in_path    = config["file"]["in_dir"]
    SOD        = config["file"]["SOD"]
    voxel_size = config["file"]["voxel_size"]
    h          = config["file"]["h"]
    w          = config["file"]["w"]
    d          = config["file"]["d"]
    gpu        = config["train"]["gpu"]

    proj_pos_path_u = '{}/fanSensorPosition_fanangle_32f.nii'.format(in_path)
    proj_pos_path_v = '{}/fanSensorPosition_coneangle_32f.nii'.format(in_path)

    device = torch.device('cuda:{}'.format(str(gpu)) if torch.cuda.is_available() else 'cpu')

    # =============================================
    # 2. Read reprojection parameters
    # =============================================
    num_angle_dense  = reproject_config["num_angle_dense"]
    model_file       = reproject_config["model_path"]
    out_path         = reproject_config["out_path"]
    out_name         = reproject_config.get("out_name", "proj_dense")
    chunk_size       = reproject_config.get("chunk_size", 100000)
    data_consistency = reproject_config.get("data_consistency", False)
    metal_mask_path  = reproject_config.get("metal_mask_path", None)
    mask_undo_y_flip = reproject_config.get("mask_undo_y_flip", True)

    # PolyNER was trained multi-energy with n_output_dims = e_level.
    # For reprojection we pick one energy channel (middle by default).
    e_level    = reproject_config.get("e_level", 6)
    energy_idx = reproject_config.get("energy_idx", int(np.mean(np.arange(0, e_level))))

    os.makedirs(out_path, exist_ok=True)

    # =============================================
    # 3. Build detector rays (angle = 0 deg)
    # =============================================
    proj_pos_u = sitk.GetArrayFromImage(sitk.ReadImage(proj_pos_path_u)).reshape(-1)
    proj_pos_v = sitk.GetArrayFromImage(sitk.ReadImage(proj_pos_path_v)).reshape(-1)
    num_det_u  = len(proj_pos_u)
    num_det_v  = len(proj_pos_v)
    num_det    = num_det_u * num_det_v

    rays = utils.cone_beam_ray(proj_pos_u, proj_pos_v, SOD)
    rays = rays.transpose(1, 0, 2, 3)                 # (num_det_v, num_det_u, 2*SOD, 3)
    num_samples_per_ray = rays.shape[2]
    rays = rays.reshape(-1, num_samples_per_ray, 3)   # (num_det, 2*SOD, 3)

    angles_dense = np.linspace(0., 360., num=num_angle_dense, endpoint=False)

    # =============================================
    # 4. Load trained model (multi-energy output)
    # =============================================
    network = tcnn.NetworkWithInputEncoding(
        n_input_dims=3, n_output_dims=e_level,
        encoding_config=config["encoding"],
        network_config=config["network"]
    ).to(device)
    network.load_state_dict(torch.load(model_file, map_location=device))
    network.eval()
    print(f"Model loaded from {model_file} (e_level={e_level}, using energy_idx={energy_idx})")

    # =============================================
    # 4b. Load metal mask (optional)
    # =============================================
    mask_tensor = lo = hi = None
    if metal_mask_path is not None:
        mask_tensor, lo, hi = load_metal_mask(
            metal_mask_path, h=h, w=w, d=d, SOD=SOD,
            device=device, undo_y_flip=mask_undo_y_flip
        )

    # =============================================
    # 5. Reprojection loop
    # =============================================
    sinogram = np.zeros((num_angle_dense, num_det), dtype=np.float32)

    with torch.no_grad():
        for idx, ang in enumerate(tqdm(angles_dense, desc="Reprojecting")):
            rays_rot = utils.rotate_ray_3d(xyz=rays.copy(), angle=ang)
            ray_flat = rays_rot.reshape(-1, 3)
            total_pts = ray_flat.shape[0]

            mu_list = []
            for s in range(0, total_pts, chunk_size):
                e = min(s + chunk_size, total_pts)
                pts = torch.from_numpy(ray_flat[s:e]).float().to(device)
                mu_chunk = network(pts)[:, energy_idx].float()  # (chunk,)

                # Skip metal: zero out mu at sample points inside the mask
                if mask_tensor is not None:
                    in_metal = points_in_metal(pts, mask_tensor, lo, hi)
                    # Debug mask alignment by temporarily using a conspicuously
                    # high fill value or inverting the mask. If the projection is
                    # still unchanged or zero, inspect points_in_metal first.
                    mu_chunk = mu_chunk.masked_fill(in_metal, 0.0)

                mu_list.append(mu_chunk.cpu().numpy())

            mu_all = np.concatenate(mu_list, axis=0)
            mu_all = mu_all.reshape(num_det, num_samples_per_ray)

            proj_line = voxel_size * np.sum(mu_all, axis=1)
            sinogram[idx] = proj_line

    sinogram = sinogram.reshape(num_angle_dense, num_det_v, num_det_u)

    # =============================================
    # 6. Data consistency (optional) -- NOT applied when masking metal,
    #    because the real sparse-view projections still contain metal.
    # =============================================
    if data_consistency:
        if metal_mask_path is not None:
            print("WARNING: data_consistency=True while metal masking is on. "
                  "Real sparse-view angles still contain metal and will "
                  "contaminate the metal-free sinogram at those indices.")
        sparse_proj_path = reproject_config["sparse_proj_path"]
        num_angle_sparse = reproject_config["num_angle_sparse"]

        proj_original = sitk.GetArrayFromImage(sitk.ReadImage(sparse_proj_path))
        proj_original = proj_original.transpose(0, 2, 1)

        scale = num_angle_dense // num_angle_sparse
        print(f"Data consistency: replacing every {scale}-th angle "
              f"({num_angle_sparse} sparse -> {num_angle_dense} dense)")

        for k in range(num_angle_sparse):
            dense_idx = k * scale
            if dense_idx < num_angle_dense:
                sinogram[dense_idx] = proj_original[k]

    # =============================================
    # 7. Save
    # =============================================
    sinogram_out = sinogram.transpose(0, 2, 1)  # (num_angle_dense, num_det_u, num_det_v)
    img_sitk = sitk.GetImageFromArray(sinogram_out)
    out_file = '{}/{}.nii'.format(out_path, out_name)
    sitk.WriteImage(img_sitk, out_file)
    print(f"Dense-view projection saved: {out_file}")
    print(f"  Shape: {sinogram_out.shape}  "
          f"(num_angle={num_angle_dense}, num_det_u={num_det_u}, num_det_v={num_det_v})")

    return sinogram


# =============================================
# Entry point
# =============================================
if __name__ == '__main__':

    with open("config.json") as f:
        config = json.load(f)

    reproject_config = {
        "num_angle_dense":  360,
        "model_path":       "./model/model_0.pkl",
        "out_path":         "./output",
        "out_name":         "proj_dense_360_metalfree",
        "chunk_size":       100000,
        "data_consistency": False,
        "sparse_proj_path": "./input/RANDO_Dose_Matching/RANDO_180vs360_views_04mm/180views_5ms/proj.nii",
        "num_angle_sparse": 180,

        # Metal masking (set metal_mask_path=None to disable)
        "metal_mask_path":  "./input/RANDO_no_implants_1mm/LE/mask.nii",
        "mask_undo_y_flip": True,

        # Network head to read out (multi-energy models have e_level>1)
        "e_level":    6,
        "energy_idx": 2,  # default = middle energy; matches test.py convention
    }

    reproject(config, reproject_config)
