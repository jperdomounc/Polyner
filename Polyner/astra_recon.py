# ----------------------------------------------#
# ASTRA reconstruction of the dense-view sinogram
# produced by reprojection.py.
#
# Polyner's forward model uses an equiangular
# (curved) detector, so before handing the
# projections to ASTRA we rebin onto a flat
# virtual detector at SDD = 2 * SOD * voxel_size.
# ----------------------------------------------#
import os
import argparse
import numpy as np
import SimpleITK as sitk
import commentjson as json

import astra


def rebin_arc_to_flat(sino_arc, theta_u_deg, theta_v_deg, SDD,
                      n_u_flat=None, n_v_flat=None):
    """
    Resample an equiangular cone-beam sinogram onto a flat virtual detector.

    Each input pixel (i, j) corresponds to a ray with fan angle theta_u[i]
    around z and cone angle theta_v[j] around x (rotations applied in that
    order, pivot at the source). For a flat detector at distance SDD from
    the source perpendicular to the central ray, the ray hits:

        u_flat = SDD * tan(theta_u) / cos(theta_v)
        v_flat = SDD * tan(theta_v)

    For each target flat pixel we invert this map and bilinearly sample
    the arc sinogram.

    Parameters
    ----------
    sino_arc : (n_angle, n_v, n_u) float32
    theta_u_deg, theta_v_deg : 1D arrays of detector angles (degrees).
    SDD : float
        Source-to-detector distance, same units as the desired output
        pixel pitch (typically mm).
    """
    n_v, n_u = len(theta_v_deg), len(theta_u_deg)
    if n_u_flat is None:
        n_u_flat = n_u
    if n_v_flat is None:
        n_v_flat = n_v

    theta_u_rad = np.deg2rad(theta_u_deg).astype(np.float64)
    theta_v_rad = np.deg2rad(theta_v_deg).astype(np.float64)

    order_u = np.argsort(theta_u_rad)
    order_v = np.argsort(theta_v_rad)
    tu_sorted = theta_u_rad[order_u]
    tv_sorted = theta_v_rad[order_v]

    u_lo = SDD * np.tan(tu_sorted[0])
    u_hi = SDD * np.tan(tu_sorted[-1])
    v_lo = SDD * np.tan(tv_sorted[0])
    v_hi = SDD * np.tan(tv_sorted[-1])
    u_new = np.linspace(u_lo, u_hi, n_u_flat)
    v_new = np.linspace(v_lo, v_hi, n_v_flat)
    pix_u = (u_hi - u_lo) / (n_u_flat - 1)
    pix_v = (v_hi - v_lo) / (n_v_flat - 1)

    UU, VV = np.meshgrid(u_new, v_new, indexing='xy')           # (n_v_flat, n_u_flat)
    theta_v_target = np.arctan2(VV, SDD)
    theta_u_target = np.arctan2(UU * np.cos(theta_v_target), SDD)

    iu = np.interp(theta_u_target, tu_sorted, np.arange(n_u, dtype=np.float64))
    iv = np.interp(theta_v_target, tv_sorted, np.arange(n_v, dtype=np.float64))

    iu0 = np.clip(np.floor(iu).astype(np.int64), 0, n_u - 2)
    iv0 = np.clip(np.floor(iv).astype(np.int64), 0, n_v - 2)
    fu = iu - iu0
    fv = iv - iv0

    sino_flat = np.empty((sino_arc.shape[0], n_v_flat, n_u_flat), dtype=np.float32)
    for k in range(sino_arc.shape[0]):
        slab = sino_arc[k][order_v][:, order_u]
        s00 = slab[iv0,     iu0]
        s10 = slab[iv0 + 1, iu0]
        s01 = slab[iv0,     iu0 + 1]
        s11 = slab[iv0 + 1, iu0 + 1]
        sino_flat[k] = ((1 - fv) * (1 - fu) * s00 +
                        fv       * (1 - fu) * s10 +
                        (1 - fv) * fu       * s01 +
                        fv       * fu       * s11).astype(np.float32)

    return sino_flat, pix_u, pix_v


def reconstruct(config, recon_config):
    in_path    = config["file"]["in_dir"]
    voxel_size = config["file"]["voxel_size"]
    SOD        = config["file"]["SOD"]
    h, w, d    = config["file"]["h"], config["file"]["w"], config["file"]["d"]

    sino_path  = recon_config["sino_path"]
    out_path   = recon_config["out_path"]
    out_name   = recon_config.get("out_name", "recon_fdk")
    algorithm  = recon_config.get("algorithm", "FDK_CUDA")
    n_iter     = recon_config.get("n_iter", 100)
    gpu_index  = recon_config.get("gpu_index", 0)

    SAD = float(SOD) * float(voxel_size)
    SDD = 2.0 * SAD
    ODD = SDD - SAD

    proj_pos_u = sitk.GetArrayFromImage(sitk.ReadImage(
        '{}/fanSensorPosition_fanangle_32f.nii'.format(in_path))).reshape(-1)
    proj_pos_v = sitk.GetArrayFromImage(sitk.ReadImage(
        '{}/fanSensorPosition_coneangle_32f.nii'.format(in_path))).reshape(-1)

    sino = sitk.GetArrayFromImage(sitk.ReadImage(sino_path)).astype(np.float32)
    n_angle, n_u_in, n_v_in = sino.shape
    print(f"Loaded sinogram {sino_path}: {sino.shape} (n_angle, n_u, n_v)")
    sino = sino.transpose(0, 2, 1)                    # (n_angle, n_v, n_u)

    assert n_u_in == len(proj_pos_u), \
        f"u mismatch: sinogram has {n_u_in}, fan-angle file has {len(proj_pos_u)}"
    assert n_v_in == len(proj_pos_v), \
        f"v mismatch: sinogram has {n_v_in}, cone-angle file has {len(proj_pos_v)}"

    sino_flat, pix_u, pix_v = rebin_arc_to_flat(
        sino, proj_pos_u, proj_pos_v, SDD,
        n_u_flat=recon_config.get("n_u_flat", n_u_in),
        n_v_flat=recon_config.get("n_v_flat", n_v_in),
    )
    print(f"Rebinned to flat detector at SDD={SDD:.4f}: "
          f"shape={sino_flat.shape}, pix_u={pix_u:.4f}, pix_v={pix_v:.4f}")

    # ASTRA wants projection volume axes (det_v, n_angle, det_u)
    sino_astra = np.ascontiguousarray(sino_flat.transpose(1, 0, 2))

    # Volume: ASTRA orders create_vol_geom args as (Y, X, Z). The returned
    # array from data3d.get is (Z, Y, X) = (slices, rows, cols).
    # We use h = X-extent, w = Y-extent, d = Z-extent (Polyner convention).
    half_x = h * voxel_size / 2.0
    half_y = w * voxel_size / 2.0
    half_z = d * voxel_size / 2.0
    vol_geom = astra.create_vol_geom(
        w, h, d,
        -half_x, half_x,
        -half_y, half_y,
        -half_z, half_z,
    )

    angles_rad = np.deg2rad(np.linspace(0., 360., num=n_angle, endpoint=False))

    proj_geom = astra.create_proj_geom(
        'cone',
        pix_u, pix_v,
        sino_astra.shape[0], sino_astra.shape[2],     # det_row_count, det_col_count
        angles_rad,
        SAD, ODD,
    )

    sino_id = astra.data3d.create('-proj3d', proj_geom, sino_astra)
    rec_id  = astra.data3d.create('-vol', vol_geom)

    cfg = astra.astra_dict(algorithm)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId']     = sino_id
    if 'CUDA' in algorithm:
        cfg['option'] = {'GPUindex': gpu_index}
    alg_id = astra.algorithm.create(cfg)

    iters = 1 if algorithm.startswith('FDK') else n_iter
    print(f"Running {algorithm} ({iters} iter)...")
    astra.algorithm.run(alg_id, iters)

    rec = astra.data3d.get(rec_id)                    # (d, w, h)
    print(f"Reconstruction returned shape {rec.shape} (d, w, h)")

    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)
    astra.data3d.delete(sino_id)

    # Match the orientation of test.py's saved volume:
    # transpose to (h, w, d) and flip along axis 1 (w / y).
    rec = np.transpose(rec, (2, 1, 0)).astype(np.float32)
    rec = np.flip(rec, axis=1).copy()

    os.makedirs(out_path, exist_ok=True)
    out_file = os.path.join(out_path, f"{out_name}.nii")
    img_sitk = sitk.GetImageFromArray(rec)
    img_sitk.SetSpacing((float(voxel_size),) * 3)
    sitk.WriteImage(img_sitk, out_file)
    print(f"Saved: {out_file}  shape={rec.shape}  spacing={voxel_size}")
    return rec


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="config.json")
    parser.add_argument("--sino",      default="./output/proj_dense_360_metalfree.nii")
    parser.add_argument("--out_dir",   default="./output")
    parser.add_argument("--out_name",  default="recon_fdk_360_metalfree")
    parser.add_argument("--algorithm", default="FDK_CUDA",
                        choices=["FDK_CUDA", "SIRT3D_CUDA", "CGLS3D_CUDA"])
    parser.add_argument("--n_iter",    type=int, default=100)
    parser.add_argument("--gpu",       type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    recon_cfg = {
        "sino_path":  args.sino,
        "out_path":   args.out_dir,
        "out_name":   args.out_name,
        "algorithm":  args.algorithm,
        "n_iter":     args.n_iter,
        "gpu_index":  args.gpu,
    }
    reconstruct(cfg, recon_cfg)
