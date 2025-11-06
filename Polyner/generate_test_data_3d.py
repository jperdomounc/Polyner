#!/usr/bin/env python3
# ----------------------------------------------#
# Pro    : cbct
# File   : generate_test_data_3d.py
# Date   : 2025
# Author : Test Data Generator for 3D CBCT
# ----------------------------------------------#
"""
Generate synthetic 3D cone-beam CT test data including:
- 3D phantom volumes (sphere, Shepp-Logan 3D)
- Forward projections
- Detector position arrays
"""

import numpy as np
import SimpleITK as sitk
import json
from pathlib import Path


def create_sphere_phantom(h, w, d, center=None, radius=None):
    """
    Create a 3D spherical phantom.

    Args:
        h, w, d: Volume dimensions
        center: Sphere center (x, y, z) in voxel coordinates. Default: volume center
        radius: Sphere radius in voxels. Default: 30% of min dimension

    Returns:
        volume: 3D array with sphere (values 0-1)
    """
    if center is None:
        center = (h // 2, w // 2, d // 2)
    if radius is None:
        radius = min(h, w, d) * 0.3

    volume = np.zeros((h, w, d), dtype=np.float32)

    # Create sphere using distance from center
    x = np.arange(h)
    y = np.arange(w)
    z = np.arange(d)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)
    volume[dist <= radius] = 1.0

    return volume


def create_ellipsoid_phantom(h, w, d):
    """
    Create a 3D ellipsoidal phantom with multiple ellipsoids.

    Returns:
        volume: 3D array with ellipsoids
    """
    volume = np.zeros((h, w, d), dtype=np.float32)

    x = np.arange(h)
    y = np.arange(w)
    z = np.arange(d)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Large ellipsoid (background)
    cx, cy, cz = h // 2, w // 2, d // 2
    a, b, c = h * 0.35, w * 0.35, d * 0.4
    ellipsoid1 = ((X - cx) / a)**2 + ((Y - cy) / b)**2 + ((Z - cz) / c)**2
    volume[ellipsoid1 <= 1] = 0.8

    # Smaller ellipsoid (higher density)
    cx2, cy2, cz2 = h * 0.5, w * 0.6, d * 0.5
    a2, b2, c2 = h * 0.15, w * 0.15, d * 0.2
    ellipsoid2 = ((X - cx2) / a2)**2 + ((Y - cy2) / b2)**2 + ((Z - cz2) / c2)**2
    volume[ellipsoid2 <= 1] = 1.0

    # Small sphere (very high density)
    cx3, cy3, cz3 = h * 0.5, w * 0.4, d * 0.4
    r3 = min(h, w, d) * 0.08
    sphere = np.sqrt((X - cx3)**2 + (Y - cy3)**2 + (Z - cz3)**2)
    volume[sphere <= r3] = 0.5

    return volume


def create_shepp_logan_3d(h, w, d):
    """
    Create a 3D Shepp-Logan phantom (simplified version).

    Returns:
        volume: 3D array with Shepp-Logan phantom
    """
    volume = np.zeros((h, w, d), dtype=np.float32)

    # Normalized coordinates [-1, 1]
    x = np.linspace(-1, 1, h)
    y = np.linspace(-1, 1, w)
    z = np.linspace(-1, 1, d)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Define ellipsoids: [center_x, center_y, center_z, axis_a, axis_b, axis_c, intensity]
    ellipsoids = [
        # Main ellipsoid (skull)
        [0.0,   0.0,    0.0,    0.69,  0.92,  0.81,  1.0],
        # Interior (brain)
        [0.0,   -0.0184, 0.0,   0.6624, 0.874, 0.78,  -0.8],
        # Right ventricle
        [0.22,  0.0,    0.0,    0.11,  0.31,  0.22,  -0.2],
        # Left ventricle
        [-0.22, 0.0,    0.0,    0.16,  0.41,  0.28,  -0.2],
        # Tumor-like structure
        [0.0,   0.35,   -0.15,  0.21,  0.25,  0.35,  0.2],
        # Small high-density
        [0.0,   0.1,    0.25,   0.046, 0.046, 0.046, 0.2],
        # Small low-density
        [0.0,   -0.1,   0.25,   0.046, 0.046, 0.046, 0.1],
        # Small structure
        [-0.08, -0.605, 0.0,    0.046, 0.023, 0.02,  0.1],
        # Small structure
        [0.0,   -0.605, 0.0,    0.023, 0.023, 0.02,  0.1],
        [0.06,  -0.605, 0.0,    0.046, 0.023, 0.02,  0.1],
    ]

    for ellipsoid in ellipsoids:
        cx, cy, cz, a, b, c, intensity = ellipsoid
        mask = ((X - cx) / a)**2 + ((Y - cy) / b)**2 + ((Z - cz) / c)**2 <= 1
        volume[mask] += intensity

    # Normalize to [0, 1]
    volume = np.clip(volume, 0, None)
    volume = volume / np.max(volume) if np.max(volume) > 0 else volume

    return volume


def forward_project_cone_beam(volume, angles, detector_u_pos, detector_v_pos,
                               SOD, SDD, voxel_size):
    """
    Simple forward projection for cone-beam CT using ray tracing.

    Args:
        volume: 3D volume (h, w, d)
        angles: Array of projection angles in degrees
        detector_u_pos: Detector horizontal positions in degrees
        detector_v_pos: Detector vertical positions in degrees
        SOD: Source-to-Origin Distance in mm
        SDD: Source-to-Detector Distance in mm
        voxel_size: Voxel size in mm

    Returns:
        projections: (num_angles, num_det_v, num_det_u) array
    """
    h, w, d = volume.shape
    num_angles = len(angles)
    num_det_u = len(detector_u_pos)
    num_det_v = len(detector_v_pos)

    projections = np.zeros((num_angles, num_det_v, num_det_u), dtype=np.float32)

    # Physical dimensions in mm
    physical_h = h * voxel_size
    physical_w = w * voxel_size
    physical_d = d * voxel_size

    print(f"Forward projecting {num_angles} angles...")

    for ia, angle in enumerate(angles):
        if ia % 20 == 0:
            print(f"  Angle {ia}/{num_angles} ({angle:.1f} deg)")

        angle_rad = np.deg2rad(angle)

        # Source position (rotates around z-axis)
        source_x = -SOD * np.sin(angle_rad)
        source_y = -SOD * np.cos(angle_rad)
        source_z = 0.0

        for iv, det_v_angle in enumerate(detector_v_pos):
            for iu, det_u_angle in enumerate(detector_u_pos):
                # Detector element position
                cone_angle_u = np.deg2rad(det_u_angle)
                cone_angle_v = np.deg2rad(det_v_angle)

                # Detector in rotated coordinate system
                det_offset_u = SDD * np.tan(cone_angle_u)
                det_offset_v = SDD * np.tan(cone_angle_v)

                # Detector position (perpendicular to source ray at SDD distance)
                det_x = source_x + (SDD / SOD) * SOD * np.sin(angle_rad) + det_offset_u * np.cos(angle_rad)
                det_y = source_y + (SDD / SOD) * SOD * np.cos(angle_rad) - det_offset_u * np.sin(angle_rad)
                det_z = source_z + det_offset_v

                # Ray direction
                ray_dir_x = det_x - source_x
                ray_dir_y = det_y - source_y
                ray_dir_z = det_z - source_z
                ray_length = np.sqrt(ray_dir_x**2 + ray_dir_y**2 + ray_dir_z**2)
                ray_dir_x /= ray_length
                ray_dir_y /= ray_length
                ray_dir_z /= ray_length

                # Sample along ray (simplified - use max dimension for num samples)
                num_samples = int(max(physical_h, physical_w, physical_d) / voxel_size * 1.5)
                line_integral = 0.0

                for step in range(num_samples):
                    t = step * voxel_size
                    px = source_x + t * ray_dir_x
                    py = source_y + t * ray_dir_y
                    pz = source_z + t * ray_dir_z

                    # Convert to voxel indices (origin at volume center)
                    vx = int((px + physical_h / 2) / voxel_size)
                    vy = int((py + physical_w / 2) / voxel_size)
                    vz = int((pz + physical_d / 2) / voxel_size)

                    # Accumulate if inside volume
                    if 0 <= vx < h and 0 <= vy < w and 0 <= vz < d:
                        line_integral += volume[vx, vy, vz]

                projections[ia, iv, iu] = line_integral * voxel_size

    return projections


def generate_test_data(config_path='./config_3d.json', output_dir='./input',
                       phantom_type='ellipsoid', num_angles=180):
    """
    Generate complete test dataset for 3D CBCT reconstruction.

    Args:
        config_path: Path to config JSON file
        output_dir: Output directory for test data
        phantom_type: Type of phantom ('sphere', 'ellipsoid', 'shepp_logan')
        num_angles: Number of projection angles
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    h = config['file']['h']
    w = config['file']['w']
    d = config['file']['d']
    voxel_size = config['file']['voxel_size']
    SOD = config['file']['SOD']
    SDD = config['file']['SDD']

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print("="*60)
    print("3D CBCT Test Data Generator")
    print("="*60)
    print(f"Volume dimensions: {h}x{w}x{d}")
    print(f"Voxel size: {voxel_size} mm")
    print(f"SOD: {SOD} mm, SDD: {SDD} mm")
    print(f"Phantom type: {phantom_type}")
    print(f"Number of angles: {num_angles}")
    print("="*60)

    # 1. Create phantom volume
    print("\n[1/4] Creating phantom volume...")
    if phantom_type == 'sphere':
        volume = create_sphere_phantom(h, w, d)
    elif phantom_type == 'ellipsoid':
        volume = create_ellipsoid_phantom(h, w, d)
    elif phantom_type == 'shepp_logan':
        volume = create_shepp_logan_3d(h, w, d)
    else:
        raise ValueError(f"Unknown phantom type: {phantom_type}")

    print(f"   Volume range: [{volume.min():.3f}, {volume.max():.3f}]")

    # Save ground truth volume
    gt_img = sitk.GetImageFromArray(volume.transpose(2, 1, 0))  # ITK uses (z, y, x)
    gt_img.SetSpacing([voxel_size, voxel_size, voxel_size])
    sitk.WriteImage(gt_img, str(output_path / 'gt_3d_0.mha'))
    print(f"   Saved: {output_path / 'gt_3d_0.mha'}")

    # 2. Create detector positions
    print("\n[2/4] Creating detector position arrays...")
    # Typical detector size in degrees (cone angle)
    # For a flat panel detector, this represents the angular coverage
    max_cone_angle = 20  # degrees (typical for CBCT)

    # Number of detector elements (should match typical detector resolution)
    num_det_u = 256  # Horizontal detector elements
    num_det_v = 192  # Vertical detector elements

    detector_u_pos = np.linspace(-max_cone_angle, max_cone_angle, num_det_u)
    detector_v_pos = np.linspace(-max_cone_angle * 0.75, max_cone_angle * 0.75, num_det_v)

    print(f"   Detector size: {num_det_u} x {num_det_v}")
    print(f"   U range: [{detector_u_pos[0]:.2f}, {detector_u_pos[-1]:.2f}] deg")
    print(f"   V range: [{detector_v_pos[0]:.2f}, {detector_v_pos[-1]:.2f}] deg")

    # Save detector positions
    det_u_img = sitk.GetImageFromArray(detector_u_pos.astype(np.float32))
    sitk.WriteImage(det_u_img, str(output_path / 'coneSensorPos_u_0.mha'))

    det_v_img = sitk.GetImageFromArray(detector_v_pos.astype(np.float32))
    sitk.WriteImage(det_v_img, str(output_path / 'coneSensorPos_v_0.mha'))

    print(f"   Saved: {output_path / 'coneSensorPos_u_0.mha'}")
    print(f"   Saved: {output_path / 'coneSensorPos_v_0.mha'}")

    # 3. Generate projection angles
    print(f"\n[3/4] Generating {num_angles} projection angles...")
    angles = np.linspace(0, 360, num_angles, endpoint=False)
    print(f"   Angle range: [0, 360) deg with {num_angles} projections")

    # 4. Forward project to create synthetic projections
    print("\n[4/4] Forward projecting (this may take a while)...")
    projections = forward_project_cone_beam(
        volume, angles, detector_u_pos, detector_v_pos,
        SOD, SDD, voxel_size
    )

    print(f"   Projection shape: {projections.shape}")
    print(f"   Projection range: [{projections.min():.3f}, {projections.max():.3f}]")

    # Normalize projections
    if projections.max() > 0:
        projections = projections / projections.max()

    # Convert to transmission (Beer-Lambert law simulation)
    # I = I0 * exp(-mu * integral)
    projections_transmission = np.exp(-projections * 2.0)  # mu ~ 2.0 for soft tissue

    # Save projections
    proj_img = sitk.GetImageFromArray(projections_transmission)
    sitk.WriteImage(proj_img, str(output_path / 'proj_3d_0.mha'))
    print(f"   Saved: {output_path / 'proj_3d_0.mha'}")

    print("\n" + "="*60)
    print("Test data generation complete!")
    print("="*60)
    print("\nGenerated files:")
    print(f"  - gt_3d_0.mha          : Ground truth volume ({h}x{w}x{d})")
    print(f"  - proj_3d_0.mha        : Projections ({num_angles}x{num_det_v}x{num_det_u})")
    print(f"  - coneSensorPos_u_0.mha: Detector U positions ({num_det_u})")
    print(f"  - coneSensorPos_v_0.mha: Detector V positions ({num_det_v})")
    print("\nNext steps:")
    print("  1. Update config_3d.json file paths if needed")
    print("  2. Update dataset_3d.py to load these files")
    print("  3. Run: python main_3d.py")
    print("="*60)

    return volume, projections


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate 3D CBCT test data')
    parser.add_argument('--config', default='./config_3d.json',
                        help='Path to config JSON')
    parser.add_argument('--output', default='./input',
                        help='Output directory')
    parser.add_argument('--phantom', default='ellipsoid',
                        choices=['sphere', 'ellipsoid', 'shepp_logan'],
                        help='Phantom type')
    parser.add_argument('--angles', type=int, default=180,
                        help='Number of projection angles')

    args = parser.parse_args()

    volume, projections = generate_test_data(
        config_path=args.config,
        output_dir=args.output,
        phantom_type=args.phantom,
        num_angles=args.angles
    )
