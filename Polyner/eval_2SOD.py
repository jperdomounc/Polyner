#!/usr/bin/env python3
"""
Evaluation script for 2SOD symmetric geometry results
Compares Polyner results with verification reconstruction
"""

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def load_nifti_as_array(file_path):
    """Load NIFTI file as numpy array."""
    if not Path(file_path).exists():
        return None
    img = sitk.ReadImage(str(file_path))
    return sitk.GetArrayFromImage(img)


def calculate_metrics(reference, test_image):
    """Calculate PSNR and SSIM metrics."""
    data_range = reference.max() - reference.min()
    
    psnr = peak_signal_noise_ratio(reference, test_image, data_range=data_range)
    ssim = structural_similarity(reference, test_image, data_range=data_range)
    
    # Calculate additional metrics
    mse = np.mean((reference - test_image) ** 2)
    mae = np.mean(np.abs(reference - test_image))
    
    return {
        'PSNR': psnr,
        'SSIM': ssim,
        'MSE': mse,
        'MAE': mae
    }


def plot_comparison(original, verification, polyner_result, output_path=None):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Top row: Images
    if original is not None:
        axes[0,0].imshow(original, cmap='gray')
        axes[0,0].set_title('Original Slice')
        axes[0,0].axis('off')
    
    if verification is not None:
        axes[0,1].imshow(verification, cmap='gray')
        axes[0,1].set_title('Verification Reconstruction')
        axes[0,1].axis('off')
    
    axes[0,2].imshow(polyner_result, cmap='gray')
    axes[0,2].set_title('Polyner Result')
    axes[0,2].axis('off')
    
    # Difference map
    if verification is not None:
        diff = np.abs(polyner_result - verification)
        im = axes[0,3].imshow(diff, cmap='hot')
        axes[0,3].set_title('|Polyner - Verification|')
        axes[0,3].axis('off')
        plt.colorbar(im, ax=axes[0,3], shrink=0.6)
    
    # Bottom row: Profiles and metrics
    center_row = polyner_result.shape[0] // 2
    
    axes[1,0].plot(polyner_result[center_row, :], 'b-', label='Polyner', linewidth=2)
    if original is not None:
        axes[1,0].plot(original[center_row, :], 'g--', label='Original', alpha=0.7)
    if verification is not None:
        axes[1,0].plot(verification[center_row, :], 'r:', label='Verification', alpha=0.7)
    axes[1,0].set_title('Horizontal Profile (Center)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    center_col = polyner_result.shape[1] // 2
    axes[1,1].plot(polyner_result[:, center_col], 'b-', label='Polyner', linewidth=2)
    if original is not None:
        axes[1,1].plot(original[:, center_col], 'g--', label='Original', alpha=0.7)
    if verification is not None:
        axes[1,1].plot(verification[:, center_col], 'r:', label='Verification', alpha=0.7)
    axes[1,1].set_title('Vertical Profile (Center)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # Histogram comparison
    if verification is not None:
        axes[1,2].hist(verification.flatten(), bins=50, alpha=0.5, label='Verification', density=True)
    axes[1,2].hist(polyner_result.flatten(), bins=50, alpha=0.5, label='Polyner', density=True)
    axes[1,2].set_title('Intensity Histograms')
    axes[1,2].legend()
    axes[1,2].set_xlabel('Intensity')
    axes[1,2].set_ylabel('Density')
    
    # Remove empty subplot
    axes[1,3].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  📊 Comparison plot saved: {output_path}")
    
    plt.show()


def evaluate_2SOD_results():
    """Evaluate 2SOD results against verification data."""
    
    parser = argparse.ArgumentParser(description='Evaluate 2SOD Polyner results')
    parser.add_argument('--config', type=str, default='config_2SOD.json',
                       help='Configuration file path')
    parser.add_argument('--img_id', type=int, default=0,
                       help='Image ID to evaluate')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save comparison plots')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    input_dir = Path(config['file']['in_dir'])
    output_dir = Path(config['file']['out_dir'])
    
    print("🎯 2SOD Symmetric Geometry Evaluation")
    print(f"   Configuration: {args.config}")
    print(f"   Image ID: {args.img_id}")
    print(f"   SOD: {config['file']['SOD']} mm")
    print(f"   SDD: {config['file']['SDD']} mm")
    
    # Load images
    print(f"\n📁 Loading images...")
    
    # Original slice
    original_path = input_dir / f"gt_{args.img_id}.nii"
    original = load_nifti_as_array(original_path)
    if original is not None:
        print(f"  ✅ Original: {original.shape}, range: [{np.min(original):.3f}, {np.max(original):.3f}]")
    else:
        print(f"  ⚠️  Original not found: {original_path}")
    
    # Verification reconstruction
    verification_path = input_dir / f"ma_{args.img_id}.nii"
    verification = load_nifti_as_array(verification_path)
    if verification is not None:
        print(f"  ✅ Verification: {verification.shape}, range: [{np.min(verification):.3f}, {np.max(verification):.3f}]")
    else:
        print(f"  ⚠️  Verification not found: {verification_path}")
    
    # Polyner result
    polyner_path = output_dir / f"polyner_{args.img_id}.nii"
    polyner_result = load_nifti_as_array(polyner_path)
    if polyner_result is not None:
        print(f"  ✅ Polyner: {polyner_result.shape}, range: [{np.min(polyner_result):.3f}, {np.max(polyner_result):.3f}]")
    else:
        print(f"  ❌ Polyner result not found: {polyner_path}")
        print("     Please run training first!")
        return
    
    # Calculate metrics
    print(f"\n📊 Quantitative Evaluation:")
    
    if original is not None:
        metrics_orig = calculate_metrics(original, polyner_result)
        print(f"\n   Polyner vs Original:")
        print(f"     PSNR: {metrics_orig['PSNR']:.2f} dB")
        print(f"     SSIM: {metrics_orig['SSIM']:.4f}")
        print(f"     MSE:  {metrics_orig['MSE']:.6f}")
        print(f"     MAE:  {metrics_orig['MAE']:.6f}")
    
    if verification is not None:
        metrics_ver = calculate_metrics(verification, polyner_result)
        print(f"\n   Polyner vs Verification:")
        print(f"     PSNR: {metrics_ver['PSNR']:.2f} dB")
        print(f"     SSIM: {metrics_ver['SSIM']:.4f}")
        print(f"     MSE:  {metrics_ver['MSE']:.6f}")
        print(f"     MAE:  {metrics_ver['MAE']:.6f}")
        
        # Additional analysis for verification
        correlation = np.corrcoef(verification.flatten(), polyner_result.flatten())[0,1]
        print(f"     Correlation: {correlation:.4f}")
    
    # Create comparison plots
    print(f"\n📈 Creating comparison plots...")
    
    plot_output_path = None
    if args.save_plots:
        plot_output_path = output_dir / f"comparison_{args.img_id}.png"
    
    plot_comparison(original, verification, polyner_result, plot_output_path)
    
    # Save metrics to file
    metrics_file = output_dir / f"metrics_{args.img_id}.json"
    metrics_data = {
        'image_id': args.img_id,
        'config': args.config,
        'geometry': {
            'SOD': config['file']['SOD'],
            'SDD': config['file']['SDD'],
            'type': config['file'].get('geometry_type', 'symmetric_fan_beam')
        }
    }
    
    if original is not None:
        metrics_data['polyner_vs_original'] = metrics_orig
    if verification is not None:
        metrics_data['polyner_vs_verification'] = metrics_ver
        metrics_data['polyner_vs_verification']['correlation'] = correlation
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    
    print(f"  💾 Metrics saved: {metrics_file}")
    
    print(f"\n✅ Evaluation complete!")
    
    return metrics_data


if __name__ == "__main__":
    evaluate_2SOD_results()