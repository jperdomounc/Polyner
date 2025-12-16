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

try:
    import astra
    print(f"ASTRA version: {astra.__version__}")
except ImportError:
    print("ERROR: ASTRA not installed. Run: pip install astra-toolbox")
    exit(1)


def main():
    # ============================================
    # GEOMETRY PARAMETERS (from config_3d.json)
    # ============================================
    SOD = 410.0        # Source-to-Origin Distance (mm)
    SDD = 620.0        # Source-to-Detector Distance (mm)
    ODD = SDD - SOD    # Origin-to-Detector Distance = 210 mm
    voxel_size = 1.0   # mm

    # Volume dimensions (from mask shape: 64, 256, 256)
    vol_x = 256        # rows
    vol_y = 256        # cols
    vol_z = 64         # slices

    print("=" * 60)
    print("ASTRA FDK Reconstruction")
    print("=" * 60)
    print(f"SOD: {SOD} mm, SDD: {SDD} mm, ODD: {ODD} mm")
    print(f"Volume: {vol_x} x {vol_y} x {vol_z} at {voxel_size} mm")
    print()

    # ============================================
    # LOAD DATA
    # ============================================
    print("Loading projection data...")
    proj_data = sitk.GetArrayFromImage(sitk.ReadImage('./input/ma_projection_0.nii'))
    print(f"  Shape: {proj_data.shape}")  # (126, 360, 334) = (det_v, angles, det_u)
    print(f"  Range: [{proj_data.min():.4f}, {proj_data.max():.4f}]")

    # Extract dimensions
    num_det_v, num_angles, num_det_u = proj_data.shape
    print(f"  num_det_v={num_det_v}, num_angles={num_angles}, num_det_u={num_det_u}")

    # Load detector positions (meshgrid format)
    print("\nLoading detector positions...")
    det_u_mesh = sitk.GetArrayFromImage(sitk.ReadImage('./input/detectorUPos.nii'))
    det_v_mesh = sitk.GetArrayFromImage(sitk.ReadImage('./input/detectorVPos.nii'))
    print(f"  Detector U mesh shape: {det_u_mesh.shape}")
    print(f"  Detector V mesh shape: {det_v_mesh.shape}")

    # Extract 1D arrays from meshgrid
    # U varies along rows (first axis of meshgrid)
    det_u_1d = det_u_mesh[:, 0]  # (334,) - horizontal angles
    # V varies along columns (second axis of meshgrid)
    det_v_1d = det_v_mesh[0, :]  # (126,) - vertical angles

    print(f"  Detector U: {len(det_u_1d)} elements, [{det_u_1d.min():.4f}, {det_u_1d.max():.4f}] deg")
    print(f"  Detector V: {len(det_v_1d)} elements, [{det_v_1d.min():.4f}, {det_v_1d.max():.4f}] deg")

    # ============================================
    # SETUP ASTRA GEOMETRY
    # ============================================
    print("\nSetting up ASTRA geometry...")

    # Projection angles (360 projections over 360 degrees)
    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

    # Calculate physical detector pixel spacing from angular spacing
    du_deg = np.abs(det_u_1d[1] - det_u_1d[0])
    dv_deg = np.abs(det_v_1d[1] - det_v_1d[0])

    # Physical spacing: SDD * tan(angle_in_radians)
    det_col_spacing = SDD * np.tan(np.deg2rad(du_deg))
    det_row_spacing = SDD * np.tan(np.deg2rad(dv_deg))

    print(f"  Angular spacing: U={du_deg:.4f}°, V={dv_deg:.4f}°")
    print(f"  Physical spacing: col={det_col_spacing:.4f} mm, row={det_row_spacing:.4f} mm")

    # Create projection geometry
    # ASTRA cone: (det_col_spacing, det_row_spacing, det_rows, det_cols, angles, SOD, ODD)
    proj_geom = astra.create_proj_geom(
        'cone',
        det_col_spacing,   # horizontal pixel spacing
        det_row_spacing,   # vertical pixel spacing
        num_det_v,         # number of detector rows
        num_det_u,         # number of detector columns
        angles,            # projection angles (radians)
        SOD,               # source-to-origin distance
        ODD                # origin-to-detector distance
    )

    # Create volume geometry (centered at origin)
    vol_size_x = vol_x * voxel_size
    vol_size_y = vol_y * voxel_size
    vol_size_z = vol_z * voxel_size

    vol_geom = astra.create_vol_geom(
        vol_x, vol_y, vol_z,
        -vol_size_x / 2, vol_size_x / 2,
        -vol_size_y / 2, vol_size_y / 2,
        -vol_size_z / 2, vol_size_z / 2
    )

    print(f"  Volume physical size: {vol_size_x} x {vol_size_y} x {vol_size_z} mm")

    # ============================================
    # PREPARE PROJECTION DATA FOR ASTRA
    # ============================================
    print("\nPreparing projection data...")

    # Your data shape: (num_det_v, num_angles, num_det_u) = (126, 360, 334)
    # ASTRA expects:   (num_det_v, num_angles, num_det_u) - SAME!
    # So we can use it directly
    proj_astra = proj_data.astype(np.float32)

    print(f"  Projection data for ASTRA: {proj_astra.shape}")

    # ============================================
    # RUN FDK RECONSTRUCTION
    # ============================================
    print("\nRunning FDK reconstruction...")

    # Create ASTRA data objects
    proj_id = astra.data3d.create('-proj3d', proj_geom, proj_astra)
    vol_id = astra.data3d.create('-vol', vol_geom)

    # Configure FDK algorithm
    cfg = astra.astra_dict('FDK_CUDA')
    cfg['ProjectionDataId'] = proj_id
    cfg['ReconstructionDataId'] = vol_id
    cfg['option'] = {'ShortScan': False}  # Full 360° scan

    # Run FDK
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    # Get reconstruction
    reconstruction = astra.data3d.get(vol_id)

    # Cleanup ASTRA objects
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(proj_id)
    astra.data3d.delete(vol_id)

    print(f"\nReconstruction complete!")
    print(f"  Shape: {reconstruction.shape}")
    print(f"  Range: [{reconstruction.min():.4f}, {reconstruction.max():.4f}]")
    print(f"  Mean: {reconstruction.mean():.4f}")

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

        # Axial slices
        z_slices = [vol_z // 4, vol_z // 2, 3 * vol_z // 4]
        for i, z in enumerate(z_slices):
            axes[0, i].imshow(reconstruction[:, :, z], cmap='gray')
            axes[0, i].set_title(f'Axial Z={z}')
            axes[0, i].axis('off')

        # Sagittal, Coronal, sample projection
        axes[1, 0].imshow(reconstruction[vol_x // 2, :, :], cmap='gray', aspect='auto')
        axes[1, 0].set_title('Sagittal')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(reconstruction[:, vol_y // 2, :], cmap='gray', aspect='auto')
        axes[1, 1].set_title('Coronal')
        axes[1, 1].axis('off')

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
