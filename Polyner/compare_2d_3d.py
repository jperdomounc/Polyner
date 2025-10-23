#!/usr/bin/env python
# ----------------------------------------------#
# Comparison script: 2D vs 3D Polyner
# Visualizes the key differences
# ----------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_geometry():
    """Visualize 2D fan-beam vs 3D cone-beam geometry"""

    fig = plt.figure(figsize=(16, 7))

    # 2D Fan-Beam Geometry
    ax1 = fig.add_subplot(121)

    # Source position
    source_2d = np.array([0, -1])

    # Arc detector
    angles = np.linspace(-30, 30, 20)
    detector_radius = 2.0
    detector_2d = np.array([
        detector_radius * np.sin(np.deg2rad(angles)),
        detector_radius * np.cos(np.deg2rad(angles))
    ])

    # Plot
    ax1.scatter(*source_2d, s=200, c='red', marker='*', label='X-ray Source', zorder=5)
    ax1.plot(detector_2d[0], detector_2d[1], 'b-', linewidth=3, label='Arc Detector')
    ax1.scatter(detector_2d[0], detector_2d[1], s=50, c='blue', zorder=4)

    # Sample rays
    for i in range(0, len(angles), 3):
        det_point = detector_2d[:, i]
        ax1.plot([source_2d[0], det_point[0]],
                [source_2d[1], det_point[1]],
                'g--', alpha=0.3, linewidth=1)

    # Reconstruction circle
    circle = plt.Circle((0, 0), 1, fill=False, linestyle='--',
                       color='gray', label='Reconstruction Region')
    ax1.add_patch(circle)

    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-1.5, 2.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_title('2D Fan-Beam CT Geometry', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    # 3D Cone-Beam Geometry
    ax2 = fig.add_subplot(122, projection='3d')

    # Source position
    source_3d = np.array([0, -1, 0])

    # Flat panel detector
    SDD_normalized = 2.0
    det_size = 1.5
    u = np.linspace(-det_size, det_size, 10)
    v = np.linspace(-det_size, det_size, 10)
    U, V = np.meshgrid(u, v)
    detector_y = np.ones_like(U) * SDD_normalized
    detector_x = U * 0.5  # Scaled for perspective
    detector_z = V * 0.5

    # Plot detector panel
    ax2.plot_surface(detector_x, detector_y, detector_z,
                    alpha=0.3, color='blue', label='Flat Panel')

    # Source
    ax2.scatter(*source_3d, s=200, c='red', marker='*',
               label='X-ray Source', zorder=5)

    # Sample rays (cone)
    for i in range(0, len(u), 3):
        for j in range(0, len(v), 3):
            det_point = np.array([detector_x[i, j], detector_y[i, j], detector_z[i, j]])
            ax2.plot([source_3d[0], det_point[0]],
                    [source_3d[1], det_point[1]],
                    [source_3d[2], det_point[2]],
                    'g--', alpha=0.2, linewidth=0.5)

    # Reconstruction sphere
    u_sphere = np.linspace(0, 2 * np.pi, 20)
    v_sphere = np.linspace(0, np.pi, 20)
    x_sphere = np.outer(np.cos(u_sphere), np.sin(v_sphere))
    y_sphere = np.outer(np.sin(u_sphere), np.sin(v_sphere))
    z_sphere = np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))
    ax2.plot_surface(x_sphere, y_sphere, z_sphere,
                    alpha=0.1, color='gray', linewidth=0)

    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-1.5, 2.5)
    ax2.set_zlim(-2, 2)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('3D Cone-Beam CT Geometry', fontsize=14, fontweight='bold')
    ax2.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig('geometry_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved geometry comparison to 'geometry_comparison.png'")


def compare_data_dimensions():
    """Print comparison table of data dimensions"""

    print("\n" + "="*80)
    print("DATA DIMENSION COMPARISON: 2D vs 3D")
    print("="*80)

    comparison = [
        ("Aspect", "2D Fan-Beam", "3D Cone-Beam"),
        ("-"*30, "-"*20, "-"*20),
        ("Network Input", "(x, y)", "(x, y, z)"),
        ("Input Dimensions", "2", "3"),
        ("Detector Config", "1D arc", "2D flat panel"),
        ("Detector Positions", "(num_det,)", "(num_det_u,), (num_det_v,)"),
        ("Projection Data", "(angles, detectors)", "(angles, det_v, det_u)"),
        ("Ray Shape", "(det, samples, 2)", "(det_v, det_u, samples, 3)"),
        ("Reconstruction Grid", "(h, w)", "(h, w, d)"),
        ("Grid Flattened", "(h*w, 2)", "(h*w*d, 3)"),
        ("Output Volume", "(h, w)", "(h, w, d)"),
        ("", "", ""),
        ("Example Sizes:", "", ""),
        ("  Detector", "611 pixels", "200×150 pixels"),
        ("  Projections", "360 angles", "360 angles"),
        ("  Volume", "1000×1000", "200×200×150"),
        ("", "", ""),
        ("Memory Usage", "~2-4 GB", "~10 GB"),
        ("Training Time", "~10 min", "~32 min"),
        ("GPU (RTX TITAN)", "✓", "✓"),
    ]

    for row in comparison:
        print(f"{row[0]:<30} {row[1]:<20} {row[2]:<20}")

    print("="*80 + "\n")


def compare_network_architecture():
    """Print network architecture comparison"""

    print("\n" + "="*80)
    print("NETWORK ARCHITECTURE COMPARISON")
    print("="*80)

    print("\n[CHANGES]")
    print("-" * 80)
    print("Parameter              2D Value          3D Value          Changed?")
    print("-" * 80)
    print(f"{'n_input_dims':<20}   {'2':<16}  {'3':<16}  {'✓ YES'}")
    print(f"{'Input coordinates':<20}   {'(x, y)':<16}  {'(x, y, z)':<16}  {'✓ YES'}")

    print("\n[UNCHANGED]")
    print("-" * 80)
    print("Parameter              Value             Notes")
    print("-" * 80)
    unchanged = [
        ("n_output_dims", "101", "Energy levels (20-120 keV)"),
        ("Encoding type", "Hash Grid", "Automatic 3D support"),
        ("n_levels", "16", "Hash grid levels"),
        ("n_features", "8", "Features per level"),
        ("Network type", "FullyFusedMLP", "Tiny-CUDA-NN"),
        ("Activation", "ReLU", "Hidden layers"),
        ("Output activation", "Squareplus", "Ensure positive μ"),
        ("n_neurons", "128", "Can increase for 3D"),
        ("n_hidden_layers", "2", "Can increase for 3D"),
    ]

    for param, value, note in unchanged:
        print(f"{param:<20}   {value:<16}  {note}")

    print("="*80 + "\n")


def compare_forward_model():
    """Show that forward model is identical"""

    print("\n" + "="*80)
    print("FORWARD MODEL COMPARISON")
    print("="*80)

    print("\nThe forward model is IDENTICAL for 2D and 3D!")
    print("-" * 80)

    code = """
    # Step 1: Network inference
    # 2D: ray coords are (batch*rays*samples, 2)
    # 3D: ray coords are (batch*rays*samples, 3)
    intensity = network(ray)  # Output: (..., energy_levels)

    # Step 2: Line integral (sum along ray)
    # Works for both 2D and 3D!
    intensity = intensity.view(batch, rays, samples, energy_levels)
    line_integral = torch.sum(intensity, dim=2)  # (batch, rays, energy_levels)

    # Step 3: Beer's Law
    transmission = torch.exp(-voxel_size * line_integral)

    # Step 4: Polyenergetic integration
    # Apply X-ray spectrum weighting
    proj = torch.sum(transmission * spectrum, dim=-1)  # (batch, rays)

    # Step 5: Log projection
    proj = -torch.log(proj)
    """

    print(code)
    print("-" * 80)
    print("KEY INSIGHT: Integration is always along the ray direction (dim=2)")
    print("             Independent of whether ray samples are 2D or 3D!")
    print("="*80 + "\n")


def main():
    """Run all comparisons"""

    print("\n" + "="*80)
    print(" POLYNER: 2D FAN-BEAM vs 3D CONE-BEAM COMPARISON")
    print("="*80)

    # 1. Data dimensions
    compare_data_dimensions()

    # 2. Network architecture
    compare_network_architecture()

    # 3. Forward model
    compare_forward_model()

    # 4. Geometry visualization
    print("\nGenerating geometry visualization...")
    visualize_geometry()

    # 5. Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
Key Takeaways:

1. CHANGES (Geometry & Network Input):
   - X-ray geometry: 2D fan-beam → 3D cone-beam
   - Network input: 2D coordinates → 3D coordinates
   - Data format: 2D projections → 3D projections

2. UNCHANGED (Everything Else):
   - Forward model (Beer's Law + polyenergetic integration)
   - Loss functions (data consistency + ASE)
   - Optimization strategy (Adam + learning rate schedule)
   - Network architecture parameters (can optionally increase for 3D)

3. PERFORMANCE:
   - Memory: 2-4 GB (2D) → ~10 GB (3D) for typical volumes
   - Time: ~10 min (2D) → ~32 min (3D) on RTX TITAN

4. BENEFIT OF 3D:
   - Local consistency along z-axis
   - True 3D metal artifact reduction
   - Better for volumetric analysis

Files created:
   - geometry_comparison.png (saved in current directory)
""")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
