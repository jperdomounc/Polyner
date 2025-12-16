#!/usr/bin/env python3
"""
Compare ASTRA FDK vs Polyner FDK Reconstructions

This script imports and runs both FDK implementations to compare them.

Run in Google Colab:
    !pip install astra-toolbox SimpleITK matplotlib tqdm
    !python compare_fdk.py
"""

import numpy as np
import SimpleITK as sitk
from pathlib import Path

# Import from other scripts
from astra_fdk_simple import load_data, run_astra_fdk, ASTRA_AVAILABLE
from polyner_fdk_simple import run_polyner_fdk


def compare_and_visualize(astra_vol, polyner_vol, output_dir='./output'):
    """Compare and visualize both reconstructions."""
    import matplotlib.pyplot as plt

    Path(output_dir).mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    print(f"\nASTRA FDK:")
    print(f"  Shape: {astra_vol.shape}")
    print(f"  Range: [{astra_vol.min():.6f}, {astra_vol.max():.6f}]")
    print(f"  Mean: {astra_vol.mean():.6f}")

    print(f"\nPolyner FDK:")
    print(f"  Shape: {polyner_vol.shape}")
    print(f"  Range: [{polyner_vol.min():.6f}, {polyner_vol.max():.6f}]")
    print(f"  Mean: {polyner_vol.mean():.6f}")

    # ASTRA returns (z, y, x), Polyner returns (x, y, z)
    # Transpose Polyner to match ASTRA for comparison
    polyner_transposed = np.transpose(polyner_vol, (2, 1, 0))  # (x,y,z) -> (z,y,x)

    print(f"\nPolyner transposed to match ASTRA: {polyner_transposed.shape}")

    if astra_vol.shape == polyner_transposed.shape:
        # Normalize both for comparison
        astra_norm = (astra_vol - astra_vol.min()) / (astra_vol.max() - astra_vol.min() + 1e-10)
        polyner_norm = (polyner_transposed - polyner_transposed.min()) / (polyner_transposed.max() - polyner_transposed.min() + 1e-10)

        diff = astra_norm - polyner_norm

        print(f"\nDifference (normalized):")
        print(f"  Mean diff: {diff.mean():.6f}")
        print(f"  Std diff: {diff.std():.6f}")
        print(f"  Max abs diff: {np.abs(diff).max():.6f}")

        corr = np.corrcoef(astra_vol.flatten(), polyner_transposed.flatten())[0, 1]
        print(f"  Correlation: {corr:.6f}")

        # Visualization
        z_dim = astra_vol.shape[0]
        z_mid = z_dim // 2

        fig, axes = plt.subplots(3, 3, figsize=(15, 15))

        # Row 1: ASTRA (z, y, x)
        axes[0, 0].imshow(astra_vol[z_mid, :, :], cmap='gray')
        axes[0, 0].set_title(f'ASTRA Axial (z={z_mid})')
        axes[0, 1].imshow(astra_vol[:, :, astra_vol.shape[2]//2], cmap='gray', aspect='auto')
        axes[0, 1].set_title('ASTRA Sagittal')
        axes[0, 2].imshow(astra_vol[:, astra_vol.shape[1]//2, :], cmap='gray', aspect='auto')
        axes[0, 2].set_title('ASTRA Coronal')

        # Row 2: Polyner (transposed to z, y, x)
        axes[1, 0].imshow(polyner_transposed[z_mid, :, :], cmap='gray')
        axes[1, 0].set_title(f'Polyner Axial (z={z_mid})')
        axes[1, 1].imshow(polyner_transposed[:, :, polyner_transposed.shape[2]//2], cmap='gray', aspect='auto')
        axes[1, 1].set_title('Polyner Sagittal')
        axes[1, 2].imshow(polyner_transposed[:, polyner_transposed.shape[1]//2, :], cmap='gray', aspect='auto')
        axes[1, 2].set_title('Polyner Coronal')

        # Row 3: Difference
        vmax = np.percentile(np.abs(diff), 99)
        axes[2, 0].imshow(diff[z_mid, :, :], cmap='RdBu', vmin=-vmax, vmax=vmax)
        axes[2, 0].set_title('Diff Axial')
        axes[2, 1].imshow(diff[:, :, diff.shape[2]//2], cmap='RdBu', vmin=-vmax, vmax=vmax, aspect='auto')
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
        print(f"\nWARNING: Shapes don't match after transpose!")
        print(f"  ASTRA: {astra_vol.shape}")
        print(f"  Polyner transposed: {polyner_transposed.shape}")

    # Save both volumes
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

    if not ASTRA_AVAILABLE:
        print("\nERROR: ASTRA not available. Install with: pip install astra-toolbox")
        return

    # Load data (using shared function from astra_fdk_simple)
    print("\nLoading data...")
    proj_data, det_u_deg, det_v_deg = load_data('./input', img_id=0)
    print(f"  Projections: {proj_data.shape}")
    print(f"  Range: [{proj_data.min():.4f}, {proj_data.max():.4f}]")

    # Run ASTRA FDK
    print("\n" + "-" * 40)
    print("Running ASTRA FDK...")
    print("-" * 40)
    astra_vol = run_astra_fdk(proj_data, det_u_deg, det_v_deg, SOD, SDD, vol_shape, voxel_size)
    print(f"ASTRA result: {astra_vol.shape}")

    # Run Polyner FDK
    print("\n" + "-" * 40)
    print("Running Polyner FDK...")
    print("-" * 40)
    polyner_vol = run_polyner_fdk(proj_data, det_u_deg, det_v_deg, SOD, SDD, vol_shape, voxel_size)
    print(f"Polyner result: {polyner_vol.shape}")

    # Compare
    compare_and_visualize(astra_vol, polyner_vol)

    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
