# ----------------------------------------------#
# ASTRA reconstruction of the dense-view sinogram
# produced by reprojection.py.
#
# Geometry summary (read from input/.../fanSensor*):
#   - Flat detector. The "fan/cone angle" files store
#     atan(pixel_pos / SDD); SDD*tan(angle) recovers
#     uniform pixel pitch.
#   - Half-fan: u angles run ~[-13.1, +0.3] deg, so the
#     detector is offset in u (centered at u_center != 0).
#     Use cone_vec, not cone.
#   - Polyner indexes proj_pos_u/v in reverse
#     (utils.cone_beam_ray: proj_pos_u[n - i - 1]),
#     so the saved sinogram axes are flipped relative
#     to the angle files; we flip them back here.
#
# SAD = SOD * voxel_size, SDD = 2 * SAD (matches the
# normalized [-1, +1] ray span in utils.cone_beam_ray).
# ----------------------------------------------#
import os
import argparse
import numpy as np
import SimpleITK as sitk
import commentjson as json

import astra


def reconstruct(config, recon_config):
    in_path    = config["file"]["in_dir"]
    voxel_size = config["file"]["voxel_size"]
    SOD        = config["file"]["SOD"]
    h, w, d    = config["file"]["h"], config["file"]["w"], config["file"]["d"]

    sino_path  = recon_config["sino_path"]
    out_path   = recon_config["out_path"]
    out_name   = recon_config.get("out_name", "recon")
    algorithm  = recon_config.get("algorithm", "SIRT3D_CUDA")
    n_iter     = recon_config.get("n_iter", 150)
    gpu_index  = recon_config.get("gpu_index", 0)

    SAD = float(SOD) * float(voxel_size)
    SDD = 2.0 * SAD
    ODD = SDD - SAD

    # =============================================
    # 1. Detector geometry from angle files
    # =============================================
    theta_u_deg = sitk.GetArrayFromImage(sitk.ReadImage(
        '{}/fanSensorPosition_fanangle_32f.nii'.format(in_path))).reshape(-1)
    theta_v_deg = sitk.GetArrayFromImage(sitk.ReadImage(
        '{}/fanSensorPosition_coneangle_32f.nii'.format(in_path))).reshape(-1)
    n_u = len(theta_u_deg)
    n_v = len(theta_v_deg)

    u_pix = SDD * np.tan(np.deg2rad(theta_u_deg))   # (n_u,) flat positions
    v_pix = SDD * np.tan(np.deg2rad(theta_v_deg))   # (n_v,)
    pitch_u = (u_pix[-1] - u_pix[0]) / (n_u - 1)
    pitch_v = (v_pix[-1] - v_pix[0]) / (n_v - 1)
    u_center = 0.5 * (u_pix[0] + u_pix[-1])         # detector u-offset
    v_center = 0.5 * (v_pix[0] + v_pix[-1])

    # Sanity: residual of "is the detector flat" assumption
    pitch_u_std = float(np.std(np.diff(u_pix)))
    pitch_v_std = float(np.std(np.diff(v_pix)))

    print(f"SAD={SAD:.4f}  SDD={SDD:.4f}  ODD={ODD:.4f}  voxel_size={voxel_size}")
    print(f"Detector: {n_u} (u) x {n_v} (v), pitch_u={pitch_u:.6f} (std {pitch_u_std:.2e}), "
          f"pitch_v={pitch_v:.6f} (std {pitch_v_std:.2e})")
    print(f"u_pix range: [{u_pix.min():.4f}, {u_pix.max():.4f}]  "
          f"v_pix range: [{v_pix.min():.4f}, {v_pix.max():.4f}]")
    print(f"detector center offset: u={u_center:.4f}, v={v_center:.4f}")

    # =============================================
    # 2. Sinogram (and undo Polyner's reversed indexing)
    # =============================================
    sino = sitk.GetArrayFromImage(sitk.ReadImage(sino_path)).astype(np.float32)
    n_angle = sino.shape[0]
    assert sino.shape == (n_angle, n_u, n_v), \
        f"sinogram shape {sino.shape} != ({n_angle}, {n_u}, {n_v})"
    print(f"Loaded sinogram: {sino.shape}  range=[{sino.min():.4g}, {sino.max():.4g}]  "
          f"mean={sino.mean():.4g}")
    if sino.max() == 0:
        print("WARNING: sinogram is all zeros — reconstruction will be empty. "
              "Re-run reprojection.py against a trained model first.")

    # Flip u and v axes so they go in ascending-angle order
    # (utils.cone_beam_ray uses proj_pos_u[n-1-i], proj_pos_v[n-1-j]).
    sino = sino[:, ::-1, ::-1].copy()

    # ASTRA 3D wants (det_row=v, n_angle, det_col=u)
    sino_astra = np.ascontiguousarray(sino.transpose(2, 0, 1))     # (n_v, n_angle, n_u)

    # =============================================
    # 3. cone_vec geometry per projection
    #
    # At angle 0:
    #   source         = (0, -SAD, 0)
    #   detector_ctr   = (u_center, +ODD, v_center)
    #   u-axis (+col)  = (pitch_u, 0, 0)
    #   v-axis (+row)  = (0, 0, pitch_v)
    # Gantry rotates source/detector CCW about z by angles_rad[k]
    # (matches utils.rotate_ray_3d's R = [[c,-s,0],[s,c,0],[0,0,1]]).
    # =============================================
    angles_rad = np.deg2rad(np.linspace(0., 360., num=n_angle, endpoint=False))
    vectors = np.zeros((n_angle, 12), dtype=np.float64)
    for k, a in enumerate(angles_rad):
        ca, sa = np.cos(a), np.sin(a)
        # rotate (x, y) by +a: (x*ca - y*sa, x*sa + y*ca)
        srcX, srcY, srcZ = (0.0)*ca - (-SAD)*sa, (0.0)*sa + (-SAD)*ca, 0.0
        dcX,  dcY,  dcZ  = u_center*ca - ODD*sa,   u_center*sa + ODD*ca, v_center
        uX,   uY,   uZ   = pitch_u*ca,             pitch_u*sa,           0.0
        vX,   vY,   vZ   = 0.0,                    0.0,                  pitch_v
        vectors[k] = [srcX, srcY, srcZ, dcX, dcY, dcZ, uX, uY, uZ, vX, vY, vZ]

    proj_geom = astra.create_proj_geom('cone_vec', n_v, n_u, vectors)

    # =============================================
    # 4. Volume geometry. Polyner convention: h=X, w=Y, d=Z.
    # ASTRA's create_vol_geom(rows=Y, cols=X, slices=Z); returned
    # array shape from data3d.get is (Z, Y, X).
    # =============================================
    half_x = h * voxel_size / 2.0
    half_y = w * voxel_size / 2.0
    half_z = d * voxel_size / 2.0
    vol_geom = astra.create_vol_geom(
        w, h, d,
        -half_x, half_x,
        -half_y, half_y,
        -half_z, half_z,
    )

    # =============================================
    # 5. Run reconstruction
    # =============================================
    sino_id = astra.data3d.create('-sino', proj_geom, sino_astra)
    rec_id  = astra.data3d.create('-vol',  vol_geom)

    cfg = astra.astra_dict(algorithm)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId']     = sino_id
    if 'CUDA' in algorithm:
        cfg['option'] = {'GPUindex': gpu_index}
    alg_id = astra.algorithm.create(cfg)

    iters = 1 if algorithm.startswith('FDK') else n_iter
    print(f"Running {algorithm} ({iters} iter)...")
    astra.algorithm.run(alg_id, iters)

    rec = astra.data3d.get(rec_id)                                # (d, w, h)
    print(f"ASTRA volume shape: {rec.shape}  range=[{rec.min():.4g}, {rec.max():.4g}]")

    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)
    astra.data3d.delete(sino_id)

    # =============================================
    # 6. Save (orient like Polyner's polyner_RANDO.nii)
    # =============================================
    rec = np.transpose(rec, (2, 1, 0)).astype(np.float32)         # (h, w, d)
    rec = np.flip(rec, axis=1).copy()

    os.makedirs(out_path, exist_ok=True)
    out_file = os.path.join(out_path, f"{out_name}.nii")
    img_sitk = sitk.GetImageFromArray(rec)
    img_sitk.SetSpacing((float(voxel_size),) * 3)
    sitk.WriteImage(img_sitk, out_file)
    print(f"Saved: {out_file}  shape={rec.shape}")
    return rec


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="config.json")
    parser.add_argument("--sino",      default="./output/proj_dense_360_metalfree.nii")
    parser.add_argument("--out_dir",   default="./output")
    parser.add_argument("--out_name",  default="recon_sirt_360_metalfree")
    parser.add_argument("--algorithm", default="SIRT3D_CUDA",
                        choices=["FDK_CUDA", "SIRT3D_CUDA", "CGLS3D_CUDA"])
    parser.add_argument("--n_iter",    type=int, default=150)
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
