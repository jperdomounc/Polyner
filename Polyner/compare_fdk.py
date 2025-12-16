#!/usr/bin/env python3
"""
Compare ASTRA FDK vs Polyner FDK Reconstructions

This script runs both reconstructions and compares them to identify
any geometry or data loading discrepancies.

Run in Google Colab:
    !pip install astra-toolbox SimpleITK matplotlib
    !python compare_fdk.py
"""

import numpy as np
import SimpleITK as sitk
from pathlib import Path

# Check for ASTRA
try:
    import astra
    ASTRA_AVAILABLE = True
    print(f"ASTRA available: {astra.__version__}")
except ImportError:
    ASTRA_AVAILABLE = False
    print("ASTRA not available - will only run Polyner FDK")


def load_data(input_dir='./input', img_id=0):
    """Load projection data."""
    input_path = Path(input_dir)

    proj_data = sitk.GetArrayFromImage(
        sitk.ReadImage(str(input_path / f'ma_projection_{img_id}.nii'))
    )

    det_u_mesh = sitk.GetArrayFromImage(
        sitk.ReadImage(str(input_path / 'detectorUPos.nii'))
    )
    det_v_mesh = sitk.GetArrayFromImage(
        sitk.ReadImage(str(input_path / 'detectorVPos.nii'))
    )

    det_u_deg = det_u_mesh[:, 0]
    det_v_deg = det_v_mesh[0, :]

    return proj_data, det_u_deg, det_v_deg


def run_astra_fdk(proj_data, det_u_deg, det_v_deg, SOD, SDD, vol_shape, voxel_size):
    """Run ASTRA FDK reconstruction."""
    if not ASTRA_AVAILABLE:
        return None

    print("\n" + "=" * 40)
    print("Running ASTRA FDK...")
    print("=" * 40)

    num_det_v, num_angles, num_det_u = proj_data.shape
    vol_x, vol_y, vol_z = vol_shape
    ODD = SDD - SOD

    # Angles
    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

    # Detector spacing
    du_deg = np.abs(det_u_deg[1] - det_u_deg[0])
    dv_deg = np.abs(det_v_deg[1] - det_v_deg[0])
    det_col_spacing = SDD * np.tan(np.deg2rad(du_deg))
    det_row_spacing = SDD * np.tan(np.deg2rad(dv_deg))

    # Geometries
    proj_geom = astra.create_proj_geom(
        'cone', det_col_spacing, det_row_spacing,
        num_det_v, num_det_u, angles, SOD, ODD
    )

    vol_geom = astra.create_vol_geom(
        vol_x, vol_y, vol_z,
        -vol_x * voxel_size / 2, vol_x * voxel_size / 2,
        -vol_y * voxel_size / 2, vol_y * voxel_size / 2,
        -vol_z * voxel_size / 2, vol_z * voxel_size / 2
    )

    # Run FDK
    proj_id = astra.data3d.create('-proj3d', proj_geom, proj_data.astype(np.float32))
    vol_id = astra.data3d.create('-vol', vol_geom)

    cfg = astra.astra_dict('FDK_CUDA')
    cfg['ProjectionDataId'] = proj_id
    cfg['ReconstructionDataId'] = vol_id
    cfg['option'] = {'ShortScan': False}

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    reconstruction = astra.data3d.get(vol_id)

    astra.algorithm.delete(alg_id)
    astra.data3d.delete(proj_id)
    astra.data3d.delete(vol_id)

    print(f"ASTRA result: {reconstruction.shape}, [{reconstruction.min():.4f}, {reconstruction.max():.4f}]")

    return reconstruction


def run_polyner_fdk_simple(proj_data, det_u_deg, det_v_deg, SOD, SDD, vol_shape, voxel_size):
    """
    Simplified Polyner-style FDK (faster, less accurate).

    This is a quick implementation for comparison purposes.
    """
    from tqdm import tqdm

    print("\n" + "=" * 40)
    print("Running Polyner FDK...")
    print("=" * 40)

    num_det_v, num_angles, num_det_u = proj_data.shape
    vol_x, vol_y, vol_z = vol_shape

    # Filter projections
    print("Filtering projections...")

    # Ramp filter
    pad_size = int(2 ** np.ceil(np.log2(2 * num_det_u)))
    freq = np.fft.fftfreq(pad_size)
    ramp = np.abs(freq) * 0.5 * (1 + np.cos(2 * np.pi * freq))  # Ram-Lak with Hann

    # Cosine weights
    cos_u = np.cos(np.deg2rad(det_u_deg))
    cos_v = np.cos(np.deg2rad(det_v_deg))
    weight = np.outer(cos_v, cos_u)

    filtered = np.zeros_like(proj_data)
    for ia in range(num_angles):
        weighted = proj_data[:, ia, :] * weight
        for iv in range(num_det_v):
            padded = np.zeros(pad_size)
            padded[:num_det_u] = weighted[iv, :]
            row_fft = np.fft.fft(padded)
            filtered[iv, ia, :] = np.fft.ifft(row_fft * ramp).real[:num_det_u]

    # Backproject
    print("Backprojecting...")

    volume = np.zeros((vol_x, vol_y, vol_z), dtype=np.float32)

    x = np.linspace(-vol_x * voxel_size / 2, vol_x * voxel_size / 2, vol_x)
    y = np.linspace(-vol_y * voxel_size / 2, vol_y * voxel_size / 2, vol_y)
    z = np.linspace(-vol_z * voxel_size / 2, vol_z * voxel_size / 2, vol_z)

    X, Y = np.meshgrid(x, y, indexing='ij')
    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

    det_u_pos = SDD * np.tan(np.deg2rad(det_u_deg))
    det_v_pos = SDD * np.tan(np.deg2rad(det_v_deg))

    for ia in tqdm(range(num_angles), desc="Backproject"):
        angle = angles[ia]
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        src_x = -SOD * sin_a
        src_y = -SOD * cos_a

        proj_slice = filtered[:, ia, :]

        for iz, vz in enumerate(z):
            dx = X - src_x
            dy = Y - src_y
            dist = np.sqrt(dx**2 + dy**2 + vz**2)

            ray_depth = dx * sin_a + dy * cos_a
            scale = SDD / (ray_depth + 1e-10)

            det_u = (dx * cos_a - dy * sin_a) * scale
            det_v = vz * scale

            u_idx = np.interp(det_u, det_u_pos, np.arange(num_det_u))
            v_idx = np.interp(det_v, det_v_pos, np.arange(num_det_v))

            u_idx = np.clip(u_idx, 0, num_det_u - 1.001)
            v_idx = np.clip(v_idx, 0, num_det_v - 1.001)

            u0, v0 = u_idx.astype(int), v_idx.astype(int)
            u1, v1 = np.minimum(u0 + 1, num_det_u - 1), np.minimum(v0 + 1, num_det_v - 1)
            wu, wv = u_idx - u0, v_idx - v0

            val = ((1-wu)*(1-wv)*proj_slice[v0, u0] + wu*(1-wv)*proj_slice[v0, u1] +
                   (1-wu)*wv*proj_slice[v1, u0] + wu*wv*proj_slice[v1, u1])

            volume[:, :, iz] += val * (SOD / dist) ** 2

    volume *= (np.pi / num_angles)

    print(f"Polyner result: {volume.shape}, [{volume.min():.4f}, {volume.max():.4f}]")

    return volume


def compare_and_visualize(astra_vol, polyner_vol, output_dir='./output'):
    """Compare and visualize both reconstructions."""
    import matplotlib.pyplot as plt

    Path(output_dir).mkdir(exist_ok=True)

    print("\n" + "=" * 40)
    print("COMPARISON RESULTS")
    print("=" * 40)

    if astra_vol is not None:
        print(f"\nASTRA FDK:")
        print(f"  Shape: {astra_vol.shape}")
        print(f"  Range: [{astra_vol.min():.6f}, {astra_vol.max():.6f}]")
        print(f"  Mean: {astra_vol.mean():.6f}")
        print(f"  Std: {astra_vol.std():.6f}")

    print(f"\nPolyner FDK:")
    print(f"  Shape: {polyner_vol.shape}")
    print(f"  Range: [{polyner_vol.min():.6f}, {polyner_vol.max():.6f}]")
    print(f"  Mean: {polyner_vol.mean():.6f}")
    print(f"  Std: {polyner_vol.std():.6f}")

    if astra_vol is not None and astra_vol.shape == polyner_vol.shape:
        # Normalize both for comparison
        astra_norm = (astra_vol - astra_vol.min()) / (astra_vol.max() - astra_vol.min() + 1e-10)
        polyner_norm = (polyner_vol - polyner_vol.min()) / (polyner_vol.max() - polyner_vol.min() + 1e-10)

        diff = astra_norm - polyner_norm

        print(f"\nDifference (normalized):")
        print(f"  Mean diff: {diff.mean():.6f}")
        print(f"  Std diff: {diff.std():.6f}")
        print(f"  Max abs diff: {np.abs(diff).max():.6f}")

        corr = np.corrcoef(astra_vol.flatten(), polyner_vol.flatten())[0, 1]
        print(f"  Correlation: {corr:.6f}")

        # Visualization
        vol_z = astra_vol.shape[2]
        z_mid = vol_z // 2

        fig, axes = plt.subplots(3, 3, figsize=(15, 15))

        # Row 1: ASTRA
        axes[0, 0].imshow(astra_vol[:, :, z_mid], cmap='gray')
        axes[0, 0].set_title('ASTRA Axial')
        axes[0, 1].imshow(astra_vol[astra_vol.shape[0]//2, :, :], cmap='gray', aspect='auto')
        axes[0, 1].set_title('ASTRA Sagittal')
        axes[0, 2].imshow(astra_vol[:, astra_vol.shape[1]//2, :], cmap='gray', aspect='auto')
        axes[0, 2].set_title('ASTRA Coronal')

        # Row 2: Polyner
        axes[1, 0].imshow(polyner_vol[:, :, z_mid], cmap='gray')
        axes[1, 0].set_title('Polyner Axial')
        axes[1, 1].imshow(polyner_vol[polyner_vol.shape[0]//2, :, :], cmap='gray', aspect='auto')
        axes[1, 1].set_title('Polyner Sagittal')
        axes[1, 2].imshow(polyner_vol[:, polyner_vol.shape[1]//2, :], cmap='gray', aspect='auto')
        axes[1, 2].set_title('Polyner Coronal')

        # Row 3: Difference
        vmax = np.percentile(np.abs(diff), 99)
        axes[2, 0].imshow(diff[:, :, z_mid], cmap='RdBu', vmin=-vmax, vmax=vmax)
        axes[2, 0].set_title('Diff Axial')
        axes[2, 1].imshow(diff[diff.shape[0]//2, :, :], cmap='RdBu', vmin=-vmax, vmax=vmax, aspect='auto')
        axes[2, 1].set_title('Diff Sagittal')
        im = axes[2, 2].imshow(diff[:, diff.shape[1]//2, :], cmap='RdBu', vmin=-vmax, vmax=vmax, aspect='auto')
        axes[2, 2].set_title('Diff Coronal')

        for ax in axes.flat:
            ax.axis('off')

        plt.colorbar(im, ax=axes[2, :], shrink=0.6, label='Normalized Difference')
        plt.suptitle(f'FDK Comparison (Correlation: {corr:.4f})', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/fdk_comparison.png', dpi=150)
        plt.show()
        print(f"\nSaved: {output_dir}/fdk_comparison.png")

    else:
        # Only Polyner available
        vol_z = polyner_vol.shape[2]
        z_mid = vol_z // 2

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(polyner_vol[:, :, z_mid], cmap='gray')
        axes[0].set_title('Axial')
        axes[1].imshow(polyner_vol[polyner_vol.shape[0]//2, :, :], cmap='gray', aspect='auto')
        axes[1].set_title('Sagittal')
        axes[2].imshow(polyner_vol[:, polyner_vol.shape[1]//2, :], cmap='gray', aspect='auto')
        axes[2].set_title('Coronal')

        for ax in axes:
            ax.axis('off')

        plt.suptitle('Polyner FDK Reconstruction', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/polyner_fdk_only.png', dpi=150)
        plt.show()

    # Save volumes
    if astra_vol is not None:
        img = sitk.GetImageFromArray(astra_vol)
        sitk.WriteImage(img, f'{output_dir}/astra_fdk.nii')
        print(f"Saved: {output_dir}/astra_fdk.nii")

    img = sitk.GetImageFromArray(polyner_vol)
    sitk.WriteImage(img, f'{output_dir}/polyner_fdk.nii')
    print(f"Saved: {output_dir}/polyner_fdk.nii")


def main():
    # Configuration
    SOD = 410.0
    SDD = 620.0
    voxel_size = 1.0
    vol_shape = (256, 256, 64)

    print("=" * 60)
    print("FDK COMPARISON: ASTRA vs Polyner")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    proj_data, det_u_deg, det_v_deg = load_data()
    print(f"Projections: {proj_data.shape}")

    # Run both FDK implementations
    astra_vol = run_astra_fdk(proj_data, det_u_deg, det_v_deg, SOD, SDD, vol_shape, voxel_size)
    polyner_vol = run_polyner_fdk_simple(proj_data, det_u_deg, det_v_deg, SOD, SDD, vol_shape, voxel_size)

    # Compare
    compare_and_visualize(astra_vol, polyner_vol)

    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
