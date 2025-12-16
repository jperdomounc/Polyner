#!/usr/bin/env python3
"""
Simple ASTRA FDK Reconstruction Script for Polyner CBCT Data

Run in Google Colab:
    1. Upload this script and your input files
    2. !pip install astra-toolbox SimpleITK
    3. Run the script

Your data format:
    - Projections: (126, 360, 334) = (num_det_v, num_angles, num_det_u)
    - Data is LINE INTEGRALS (not transmission) - no -log needed
    - Detector positions: meshgrid (334, 126), angles in degrees
"""

import numpy as np
import SimpleITK as sitk
from pathlib import Path

# Check ASTRA availability
ASTRA_AVAILABLE = False
try:
    import astra
    ASTRA_AVAILABLE = True
except ImportError:
    pass


def load_data(input_dir='./input', img_id=0):
    """
    Load projection data using Polyner's conventions.

    Returns:
        proj_data: (num_det_v, num_angles, num_det_u) projection data
        det_u_deg: 1D array of detector U angles (degrees)
        det_v_deg: 1D array of detector V angles (degrees)
    """
    input_path = Path(input_dir)

    # Load projections
    proj_path = input_path / f'ma_projection_{img_id}.nii'
    proj_data = sitk.GetArrayFromImage(sitk.ReadImage(str(proj_path)))

    # Load detector positions (meshgrid format)
    det_u_mesh = sitk.GetArrayFromImage(sitk.ReadImage(str(input_path / 'detectorUPos.nii')))
    det_v_mesh = sitk.GetArrayFromImage(sitk.ReadImage(str(input_path / 'detectorVPos.nii')))

    # Extract 1D arrays (U varies along rows, V varies along cols of meshgrid)
    det_u_deg = det_u_mesh[:, 0]   # (334,) horizontal fan angles
    det_v_deg = det_v_mesh[0, :]   # (126,) vertical cone angles

    return proj_data, det_u_deg, det_v_deg


def run_astra_fdk(proj_data, det_u_deg, det_v_deg, SOD, SDD, vol_shape, voxel_size):
    """
    Run ASTRA FDK reconstruction.

    Args:
        proj_data: (num_det_v, num_angles, num_det_u) projection data
        det_u_deg: 1D array of detector U angles (degrees)
        det_v_deg: 1D array of detector V angles (degrees)
        SOD: Source-to-Origin distance (mm)
        SDD: Source-to-Detector distance (mm)
        vol_shape: (vol_x, vol_y, vol_z) volume dimensions
        voxel_size: voxel size in mm

    Returns:
        reconstruction: 3D numpy array (z, y, x)
    """
    if not ASTRA_AVAILABLE:
        raise ImportError("ASTRA not installed. Run: pip install astra-toolbox")

    num_det_v, num_angles, num_det_u = proj_data.shape
    vol_x, vol_y, vol_z = vol_shape
    ODD = SDD - SOD

    # Projection angles (radians)
    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

    # Calculate physical detector pixel spacing from angular spacing
    du_deg = np.abs(det_u_deg[1] - det_u_deg[0])
    dv_deg = np.abs(det_v_deg[1] - det_v_deg[0])
    det_col_spacing = SDD * np.tan(np.deg2rad(du_deg))
    det_row_spacing = SDD * np.tan(np.deg2rad(dv_deg))

    # Create projection geometry
    proj_geom = astra.create_proj_geom(
        'cone',
        det_col_spacing,
        det_row_spacing,
        num_det_v,
        num_det_u,
        angles,
        SOD,
        ODD
    )

    # Create volume geometry (centered at origin)
    vol_geom = astra.create_vol_geom(
        vol_x, vol_y, vol_z,
        -vol_x * voxel_size / 2, vol_x * voxel_size / 2,
        -vol_y * voxel_size / 2, vol_y * voxel_size / 2,
        -vol_z * voxel_size / 2, vol_z * voxel_size / 2
    )

    # Create ASTRA data objects
    proj_id = astra.data3d.create('-proj3d', proj_geom, proj_data.astype(np.float32))
    vol_id = astra.data3d.create('-vol', vol_geom)

    # Configure and run FDK
    cfg = astra.astra_dict('FDK_CUDA')
    cfg['ProjectionDataId'] = proj_id
    cfg['ReconstructionDataId'] = vol_id
    cfg['option'] = {'ShortScan': False}

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    # Get reconstruction
    reconstruction = astra.data3d.get(vol_id)

    # Cleanup
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(proj_id)
    astra.data3d.delete(vol_id)

    return reconstruction


def main():
    # ============================================
    # GEOMETRY PARAMETERS (from config_3d.json)
    # ============================================
    SOD = 410.0        # Source-to-Origin Distance (mm)
    SDD = 620.0        # Source-to-Detector Distance (mm)
    voxel_size = 1.0   # mm

    # Volume dimensions
    vol_shape = (256, 256, 64)  # (x, y, z)

    if not ASTRA_AVAILABLE:
        print("ERROR: ASTRA not installed. Run: pip install astra-toolbox")
        return None

    print("=" * 60)
    print("ASTRA FDK Reconstruction")
    print("=" * 60)
    print(f"ASTRA version: {astra.__version__}")
    print(f"SOD: {SOD} mm, SDD: {SDD} mm, ODD: {SDD - SOD} mm")
    print(f"Volume: {vol_shape[0]} x {vol_shape[1]} x {vol_shape[2]} at {voxel_size} mm")
    print()

    # ============================================
    # LOAD DATA
    # ============================================
    print("Loading data...")
    proj_data, det_u_deg, det_v_deg = load_data('./input', img_id=0)
    print(f"  Projections: {proj_data.shape}")
    print(f"  Range: [{proj_data.min():.4f}, {proj_data.max():.4f}]")
    print(f"  Detector U: {len(det_u_deg)} elements, [{det_u_deg.min():.4f}, {det_u_deg.max():.4f}] deg")
    print(f"  Detector V: {len(det_v_deg)} elements, [{det_v_deg.min():.4f}, {det_v_deg.max():.4f}] deg")

    # ============================================
    # RUN FDK RECONSTRUCTION
    # ============================================
    print("\nRunning FDK reconstruction...")
    reconstruction = run_astra_fdk(proj_data, det_u_deg, det_v_deg, SOD, SDD, vol_shape, voxel_size)

    print(f"\nReconstruction complete!")
    print(f"  Shape: {reconstruction.shape}")
    print(f"  Range: [{reconstruction.min():.4f}, {reconstruction.max():.4f}]")
    print(f"  Mean: {reconstruction.mean():.4f}")

    # ASTRA returns volume in (z, y, x) order
    # Get actual dimensions from output
    out_z, out_y, out_x = reconstruction.shape
    print(f"  ASTRA output order: (z={out_z}, y={out_y}, x={out_x})")

    # ============================================
    # SAVE OUTPUT
    # ============================================
    output_file = './output/astra_fdk_reconstruction.nii'

    img = sitk.GetImageFromArray(reconstruction)
    img.SetSpacing([voxel_size, voxel_size, voxel_size])
    sitk.WriteImage(img, output_file)

    print(f"\nSaved: {output_file}")

    # ============================================
    # VISUALIZATION (optional - for Colab)
    # ============================================
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # ASTRA output is (z, y, x)
        # Axial slices (fixed z, show x-y plane)
        z_slices = [out_z // 4, out_z // 2, 3 * out_z // 4]
        for i, z in enumerate(z_slices):
            axes[0, i].imshow(reconstruction[z, :, :], cmap='gray')
            axes[0, i].set_title(f'Axial Z={z}')
            axes[0, i].axis('off')

        # Sagittal (fixed x, show z-y plane)
        axes[1, 0].imshow(reconstruction[:, :, out_x // 2], cmap='gray', aspect='auto')
        axes[1, 0].set_title(f'Sagittal X={out_x // 2}')
        axes[1, 0].axis('off')

        # Coronal (fixed y, show z-x plane)
        axes[1, 1].imshow(reconstruction[:, out_y // 2, :], cmap='gray', aspect='auto')
        axes[1, 1].set_title(f'Coronal Y={out_y // 2}')
        axes[1, 1].axis('off')

        # Sample projection
        axes[1, 2].imshow(proj_data[:, 0, :], cmap='gray')
        axes[1, 2].set_title('Projection @ angle 0')
        axes[1, 2].axis('off')

        plt.suptitle('ASTRA FDK Reconstruction', fontsize=14)
        plt.tight_layout()
        plt.savefig('./output/astra_fdk_visualization.png', dpi=150)
        plt.show()
        print("Saved visualization: ./output/astra_fdk_visualization.png")

    except ImportError:
        print("Matplotlib not available - skipping visualization")

    return reconstruction


if __name__ == '__main__':
    reconstruction = main()
