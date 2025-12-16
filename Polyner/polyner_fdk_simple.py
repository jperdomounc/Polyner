#!/usr/bin/env python3
"""
Polyner FDK Reconstruction (No Deep Learning)

This implements FDK reconstruction using Polyner's data loading and geometry conventions.
Use this to compare against ASTRA FDK to validate data loading.

FDK Algorithm:
1. Weight projections by cosine of cone angle
2. Filter each detector row with ramp filter
3. Backproject onto 3D volume
"""

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from pathlib import Path


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
    print(f"Projections: {proj_data.shape}")  # (126, 360, 334) = (det_v, angles, det_u)
    print(f"  Range: [{proj_data.min():.4f}, {proj_data.max():.4f}]")

    # Load detector positions (meshgrid format)
    det_u_mesh = sitk.GetArrayFromImage(sitk.ReadImage(str(input_path / 'detectorUPos.nii')))
    det_v_mesh = sitk.GetArrayFromImage(sitk.ReadImage(str(input_path / 'detectorVPos.nii')))

    # Extract 1D arrays (U varies along rows, V varies along cols of meshgrid)
    det_u_deg = det_u_mesh[:, 0]   # (334,) horizontal fan angles
    det_v_deg = det_v_mesh[0, :]   # (126,) vertical cone angles

    print(f"Detector U: {len(det_u_deg)} elements, [{det_u_deg.min():.2f}, {det_u_deg.max():.2f}] deg")
    print(f"Detector V: {len(det_v_deg)} elements, [{det_v_deg.min():.2f}, {det_v_deg.max():.2f}] deg")

    return proj_data, det_u_deg, det_v_deg


def create_ramp_filter(num_cols):
    """
    Create ramp filter for FDK in frequency domain.

    The ramp filter is |omega| in frequency domain.
    We use Ram-Lak filter with some smoothing.
    """
    # Pad to next power of 2 for efficient FFT
    pad_size = int(2 ** np.ceil(np.log2(2 * num_cols)))

    # Frequency axis
    freq = np.fft.fftfreq(pad_size)

    # Ramp filter = |frequency|
    ramp = np.abs(freq)

    # Apply Hann window for smoothing (reduces ringing artifacts)
    hann = 0.5 * (1 + np.cos(2 * np.pi * freq))
    ramp_filtered = ramp * hann

    return ramp_filtered, pad_size


def filter_projections(proj_data, det_u_deg, det_v_deg, SDD):
    """
    Apply FDK weighting and ramp filtering to projections.

    Steps:
    1. Weight by cosine of cone angles (FDK weighting)
    2. Apply ramp filter to each row

    Args:
        proj_data: (num_det_v, num_angles, num_det_u) projections
        det_u_deg: horizontal detector angles (degrees)
        det_v_deg: vertical detector angles (degrees)
        SDD: source-to-detector distance

    Returns:
        filtered: filtered projections same shape as input
    """
    num_det_v, num_angles, num_det_u = proj_data.shape

    print("\nFiltering projections...")

    # Convert angles to radians
    det_u_rad = np.deg2rad(det_u_deg)
    det_v_rad = np.deg2rad(det_v_deg)

    # FDK cosine weighting
    # Weight = cos(gamma) * cos(kappa) where gamma=fan angle, kappa=cone angle
    # Create 2D weight array (det_v, det_u)
    cos_u = np.cos(det_u_rad)  # (num_det_u,)
    cos_v = np.cos(det_v_rad)  # (num_det_v,)
    weight = np.outer(cos_v, cos_u)  # (num_det_v, num_det_u)

    # Create ramp filter
    ramp_filter, pad_size = create_ramp_filter(num_det_u)

    # Apply weighting and filtering
    filtered = np.zeros_like(proj_data)

    for angle_idx in tqdm(range(num_angles), desc="Filtering"):
        proj_slice = proj_data[:, angle_idx, :]  # (num_det_v, num_det_u)

        # Apply cosine weight
        weighted = proj_slice * weight

        # Filter each row (along detector U direction)
        for v_idx in range(num_det_v):
            row = weighted[v_idx, :]

            # Pad for FFT
            padded = np.zeros(pad_size)
            padded[:num_det_u] = row

            # Apply ramp filter in frequency domain
            row_fft = np.fft.fft(padded)
            row_filtered = np.fft.ifft(row_fft * ramp_filter).real

            # Extract filtered row
            filtered[v_idx, angle_idx, :] = row_filtered[:num_det_u]

    print(f"  Filtered range: [{filtered.min():.4f}, {filtered.max():.4f}]")

    return filtered


def backproject_fdk(filtered_proj, det_u_deg, det_v_deg,
                    SOD, SDD, vol_shape, voxel_size):
    """
    Backproject filtered projections onto 3D volume.

    Uses Polyner's coordinate conventions:
    - Source rotates around Z axis
    - Detector is perpendicular to source-origin line

    Args:
        filtered_proj: (num_det_v, num_angles, num_det_u) filtered projections
        det_u_deg: horizontal detector angles (degrees)
        det_v_deg: vertical detector angles (degrees)
        SOD: source-to-origin distance (mm)
        SDD: source-to-detector distance (mm)
        vol_shape: (vol_x, vol_y, vol_z) volume dimensions
        voxel_size: voxel size in mm

    Returns:
        volume: reconstructed 3D volume
    """
    num_det_v, num_angles, num_det_u = filtered_proj.shape
    vol_x, vol_y, vol_z = vol_shape

    print(f"\nBackprojecting to {vol_x}x{vol_y}x{vol_z} volume...")

    # Initialize volume
    volume = np.zeros((vol_x, vol_y, vol_z), dtype=np.float32)

    # Create volume coordinates (centered at origin)
    x = np.linspace(-vol_x * voxel_size / 2, vol_x * voxel_size / 2, vol_x)
    y = np.linspace(-vol_y * voxel_size / 2, vol_y * voxel_size / 2, vol_y)
    z = np.linspace(-vol_z * voxel_size / 2, vol_z * voxel_size / 2, vol_z)

    # Projection angles (360 projections over 360 degrees)
    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

    # Convert detector angles to physical positions at detector
    # Using small angle: position = SDD * tan(angle)
    det_u_pos = SDD * np.tan(np.deg2rad(det_u_deg))  # mm
    det_v_pos = SDD * np.tan(np.deg2rad(det_v_deg))  # mm

    # Detector spacing for interpolation
    du = det_u_pos[1] - det_u_pos[0] if len(det_u_pos) > 1 else 1.0
    dv = det_v_pos[1] - det_v_pos[0] if len(det_v_pos) > 1 else 1.0

    # Backproject each angle
    for angle_idx in tqdm(range(num_angles), desc="Backprojecting"):
        angle = angles[angle_idx]

        # Source position
        src_x = -SOD * np.sin(angle)
        src_y = -SOD * np.cos(angle)
        src_z = 0.0

        # Detector center position
        det_cx = (SDD - SOD) * np.sin(angle)
        det_cy = (SDD - SOD) * np.cos(angle)
        det_cz = 0.0

        # Detector coordinate system
        # u-axis: perpendicular to source-detector line, in XY plane
        u_axis = np.array([np.cos(angle), -np.sin(angle), 0])
        # v-axis: vertical (Z direction)
        v_axis = np.array([0, 0, 1])

        # Get filtered projection for this angle
        proj_slice = filtered_proj[:, angle_idx, :]  # (num_det_v, num_det_u)

        # For each voxel, compute detector coordinates and accumulate
        for iz, vz in enumerate(z):
            for iy, vy in enumerate(y):
                for ix, vx in enumerate(x):
                    # Vector from source to voxel
                    voxel_vec = np.array([vx - src_x, vy - src_y, vz - src_z])

                    # Distance from source to voxel along ray
                    dist_src_voxel = np.sqrt(np.sum(voxel_vec**2))

                    # Normalize ray direction
                    ray_dir = voxel_vec / dist_src_voxel

                    # Find intersection with detector plane
                    # Detector plane: point on plane is (det_cx, det_cy, det_cz)
                    # Normal to plane points from origin to detector
                    det_normal = np.array([np.sin(angle), np.cos(angle), 0])

                    # Ray-plane intersection parameter
                    denom = np.dot(ray_dir, det_normal)
                    if np.abs(denom) < 1e-10:
                        continue

                    t = np.dot(np.array([det_cx, det_cy, det_cz]) - np.array([src_x, src_y, src_z]), det_normal) / denom

                    if t < 0:  # Behind source
                        continue

                    # Intersection point on detector
                    hit_point = np.array([src_x, src_y, src_z]) + t * ray_dir

                    # Convert to detector coordinates (relative to detector center)
                    hit_rel = hit_point - np.array([det_cx, det_cy, det_cz])
                    det_u = np.dot(hit_rel, u_axis)  # horizontal position
                    det_v = np.dot(hit_rel, v_axis)  # vertical position

                    # Convert to pixel indices
                    u_idx = (det_u - det_u_pos[0]) / du
                    v_idx = (det_v - det_v_pos[0]) / dv

                    # Bilinear interpolation bounds check
                    if u_idx < 0 or u_idx >= num_det_u - 1:
                        continue
                    if v_idx < 0 or v_idx >= num_det_v - 1:
                        continue

                    # Bilinear interpolation
                    u0, v0 = int(u_idx), int(v_idx)
                    u1, v1 = u0 + 1, v0 + 1
                    wu = u_idx - u0
                    wv = v_idx - v0

                    val = (1 - wu) * (1 - wv) * proj_slice[v0, u0] + \
                          wu * (1 - wv) * proj_slice[v0, u1] + \
                          (1 - wu) * wv * proj_slice[v1, u0] + \
                          wu * wv * proj_slice[v1, u1]

                    # FDK distance weighting: (SOD / distance_to_source)^2
                    weight = (SOD / dist_src_voxel) ** 2

                    volume[ix, iy, iz] += val * weight

    # Normalize by number of angles
    volume *= (np.pi / num_angles)

    print(f"  Volume range: [{volume.min():.4f}, {volume.max():.4f}]")

    return volume


def backproject_fdk_fast(filtered_proj, det_u_deg, det_v_deg,
                         SOD, SDD, vol_shape, voxel_size):
    """
    Faster vectorized FDK backprojection.

    Processes one Z-slice at a time with vectorized XY operations.
    """
    num_det_v, num_angles, num_det_u = filtered_proj.shape
    vol_x, vol_y, vol_z = vol_shape

    print(f"\nBackprojecting to {vol_x}x{vol_y}x{vol_z} volume (fast mode)...")

    # Initialize volume
    volume = np.zeros((vol_x, vol_y, vol_z), dtype=np.float32)

    # Create volume coordinates (centered at origin)
    x = np.linspace(-vol_x * voxel_size / 2, vol_x * voxel_size / 2, vol_x)
    y = np.linspace(-vol_y * voxel_size / 2, vol_y * voxel_size / 2, vol_y)
    z = np.linspace(-vol_z * voxel_size / 2, vol_z * voxel_size / 2, vol_z)

    # Create meshgrid for XY plane
    X, Y = np.meshgrid(x, y, indexing='ij')  # (vol_x, vol_y)

    # Projection angles
    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

    # Detector positions (physical)
    det_u_pos = SDD * np.tan(np.deg2rad(det_u_deg))
    det_v_pos = SDD * np.tan(np.deg2rad(det_v_deg))

    ODD = SDD - SOD  # Origin-to-detector distance

    # Process each angle
    for angle_idx in tqdm(range(num_angles), desc="Backprojecting"):
        angle = angles[angle_idx]
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # Source position
        src_x = -SOD * sin_a
        src_y = -SOD * cos_a

        # Get filtered projection
        proj_slice = filtered_proj[:, angle_idx, :]  # (num_det_v, num_det_u)

        # Process each Z slice
        for iz, vz in enumerate(z):
            # Vector from source to each voxel in XY plane
            dx = X - src_x  # (vol_x, vol_y)
            dy = Y - src_y
            dz = vz  # scalar for this slice

            # Distance from source to each voxel
            dist = np.sqrt(dx**2 + dy**2 + dz**2)

            # Project voxel onto detector
            # Using similar triangles: det_coord = (SDD / ray_depth) * offset
            # Ray depth = projection onto source-detector axis
            ray_depth = dx * sin_a + dy * cos_a  # distance along central ray

            # Detector coordinates
            # u = horizontal offset at detector
            # v = vertical offset at detector
            scale = SDD / (ray_depth + 1e-10)

            # Horizontal: perpendicular to central ray in XY plane
            det_u = (dx * cos_a - dy * sin_a) * scale
            # Vertical: z offset scaled to detector
            det_v = dz * scale

            # Convert to pixel indices
            u_idx = np.interp(det_u, det_u_pos, np.arange(num_det_u))
            v_idx = np.interp(det_v, det_v_pos, np.arange(num_det_v))

            # Clip to valid range
            u_idx = np.clip(u_idx, 0, num_det_u - 1.001)
            v_idx = np.clip(v_idx, 0, num_det_v - 1.001)

            # Bilinear interpolation
            u0 = u_idx.astype(int)
            v0 = v_idx.astype(int)
            u1 = np.minimum(u0 + 1, num_det_u - 1)
            v1 = np.minimum(v0 + 1, num_det_v - 1)

            wu = u_idx - u0
            wv = v_idx - v0

            # Sample projection values
            val = (1 - wu) * (1 - wv) * proj_slice[v0, u0] + \
                  wu * (1 - wv) * proj_slice[v0, u1] + \
                  (1 - wu) * wv * proj_slice[v1, u0] + \
                  wu * wv * proj_slice[v1, u1]

            # FDK weighting
            weight = (SOD / dist) ** 2

            # Accumulate
            volume[:, :, iz] += val * weight

    # Normalize
    volume *= (np.pi / num_angles)

    print(f"  Volume range: [{volume.min():.4f}, {volume.max():.4f}]")

    return volume


def run_polyner_fdk(proj_data, det_u_deg, det_v_deg, SOD, SDD, vol_shape, voxel_size):
    """
    Run Polyner-style FDK reconstruction.

    Args:
        proj_data: (num_det_v, num_angles, num_det_u) projection data
        det_u_deg: 1D array of detector U angles (degrees)
        det_v_deg: 1D array of detector V angles (degrees)
        SOD: Source-to-Origin distance (mm)
        SDD: Source-to-Detector distance (mm)
        vol_shape: (vol_x, vol_y, vol_z) volume dimensions
        voxel_size: voxel size in mm

    Returns:
        reconstruction: 3D numpy array (x, y, z)
    """
    # Step 1: Filter projections
    filtered = filter_projections(proj_data, det_u_deg, det_v_deg, SDD)

    # Step 2: Backproject
    volume = backproject_fdk_fast(
        filtered, det_u_deg, det_v_deg,
        SOD, SDD, vol_shape, voxel_size
    )

    return volume


def main():
    """Main function to run Polyner-style FDK reconstruction."""

    # ============================================
    # CONFIGURATION (from config_3d.json)
    # ============================================
    input_dir = './input'
    output_dir = './output'

    SOD = 410.0        # Source-to-Origin Distance (mm)
    SDD = 620.0        # Source-to-Detector Distance (mm)
    voxel_size = 1.0   # mm

    # Volume dimensions
    vol_shape = (256, 256, 64)

    print("=" * 60)
    print("Polyner FDK Reconstruction (No Deep Learning)")
    print("=" * 60)
    print(f"SOD: {SOD} mm, SDD: {SDD} mm")
    print(f"Volume: {vol_shape[0]} x {vol_shape[1]} x {vol_shape[2]} at {voxel_size} mm")
    print()

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # ============================================
    # LOAD DATA
    # ============================================
    proj_data, det_u_deg, det_v_deg = load_data(input_dir)
    print(f"Projections: {proj_data.shape}")
    print(f"  Range: [{proj_data.min():.4f}, {proj_data.max():.4f}]")

    # ============================================
    # FDK RECONSTRUCTION
    # ============================================
    volume = run_polyner_fdk(proj_data, det_u_deg, det_v_deg, SOD, SDD, vol_shape, voxel_size)

    print(f"\nReconstruction complete!")
    print(f"  Shape: {volume.shape}")
    print(f"  Range: [{volume.min():.4f}, {volume.max():.4f}]")

    # ============================================
    # SAVE OUTPUT
    # ============================================
    output_file = f'{output_dir}/polyner_fdk_reconstruction.nii'

    img = sitk.GetImageFromArray(volume)
    img.SetSpacing([voxel_size, voxel_size, voxel_size])
    sitk.WriteImage(img, output_file)

    print(f"\nSaved: {output_file}")

    # ============================================
    # VISUALIZATION
    # ============================================
    try:
        import matplotlib.pyplot as plt

        vol_x, vol_y, vol_z = vol_shape

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Axial slices
        z_slices = [vol_z // 4, vol_z // 2, 3 * vol_z // 4]
        for i, z in enumerate(z_slices):
            axes[0, i].imshow(volume[:, :, z], cmap='gray')
            axes[0, i].set_title(f'Axial Z={z}')
            axes[0, i].axis('off')

        # Other views
        axes[1, 0].imshow(volume[vol_x // 2, :, :], cmap='gray', aspect='auto')
        axes[1, 0].set_title('Sagittal')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(volume[:, vol_y // 2, :], cmap='gray', aspect='auto')
        axes[1, 1].set_title('Coronal')
        axes[1, 1].axis('off')

        # Sample projection
        axes[1, 2].imshow(proj_data[:, 0, :], cmap='gray')
        axes[1, 2].set_title('Projection @ angle 0')
        axes[1, 2].axis('off')

        plt.suptitle('Polyner FDK Reconstruction (No DL)', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/polyner_fdk_visualization.png', dpi=150)
        plt.show()
        print(f"Saved: {output_dir}/polyner_fdk_visualization.png")

    except ImportError:
        print("Matplotlib not available - skipping visualization")

    return volume


if __name__ == '__main__':
    volume = main()
