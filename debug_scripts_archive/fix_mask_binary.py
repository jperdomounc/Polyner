import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

def fix_mask_to_binary(mask_path, threshold=0.5, visualize=True):
    """Convert mask to proper binary format (0 and 1)"""
    
    print(f"=== FIXING MASK TO BINARY FORMAT ===")
    
    # Load mask
    sitk_img = sitk.ReadImage(mask_path)
    mask = sitk.GetArrayFromImage(sitk_img)
    
    print(f"Original mask:")
    print(f"  Shape: {mask.shape}")
    print(f"  Range: [{mask.min():.6f}, {mask.max():.6f}]")
    print(f"  Unique values (first 10): {np.unique(mask)[:10]}")
    print(f"  Mean: {mask.mean():.6f}")
    
    # Convert to binary using threshold
    binary_mask = (mask > threshold).astype(np.float32)
    
    print(f"\\nAfter binary conversion (threshold={threshold}):")
    print(f"  Shape: {binary_mask.shape}")
    print(f"  Range: [{binary_mask.min():.6f}, {binary_mask.max():.6f}]")
    print(f"  Unique values: {np.unique(binary_mask)}")
    print(f"  Metal pixels (1): {np.sum(binary_mask == 1)} ({100*np.sum(binary_mask == 1)/binary_mask.size:.2f}%)")
    print(f"  Background pixels (0): {np.sum(binary_mask == 0)} ({100*np.sum(binary_mask == 0)/binary_mask.size:.2f}%)")
    
    if visualize:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original mask
        im1 = axes[0].imshow(mask, cmap='gray', origin='lower')
        axes[0].set_title('Original Mask (Non-binary)')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[0])
        
        # Binary mask
        im2 = axes[1].imshow(binary_mask, cmap='gray', origin='lower')
        axes[1].set_title(f'Fixed Binary Mask (threshold={threshold})')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        plt.colorbar(im2, ax=axes[1])
        
        # Difference
        diff = np.abs(mask - binary_mask)
        im3 = axes[2].imshow(diff, cmap='hot', origin='lower')
        axes[2].set_title('Difference (|Original - Binary|)')
        axes[2].set_xlabel('X')
        axes[2].set_ylabel('Y')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig('mask_binary_fix.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ Visualization saved as 'mask_binary_fix.png'")
        plt.show()
    
    # Save fixed mask
    output_img = sitk.GetImageFromArray(binary_mask)
    output_img.SetSpacing(sitk_img.GetSpacing())
    output_img.SetOrigin(sitk_img.GetOrigin())
    output_img.SetDirection(sitk_img.GetDirection())
    
    sitk.WriteImage(output_img, mask_path)
    print(f"  ✓ Fixed binary mask saved to {mask_path}")
    
    return binary_mask

def test_different_thresholds(mask_path):
    """Test different threshold values to find optimal binarization"""
    
    sitk_img = sitk.ReadImage(mask_path)
    mask = sitk.GetArrayFromImage(sitk_img)
    
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    print(f"\\n=== TESTING DIFFERENT THRESHOLDS ===")
    print(f"Original mask range: [{mask.min():.3f}, {mask.max():.3f}]")
    print(f"Original mask mean: {mask.mean():.3f}")
    
    fig, axes = plt.subplots(1, len(thresholds), figsize=(20, 4))
    
    results = []
    for i, thresh in enumerate(thresholds):
        binary = (mask > thresh).astype(np.float32)
        metal_percent = 100 * np.sum(binary == 1) / binary.size
        
        results.append({
            'threshold': thresh,
            'metal_percent': metal_percent,
            'metal_pixels': np.sum(binary == 1)
        })
        
        axes[i].imshow(binary, cmap='gray', origin='lower')
        axes[i].set_title(f'Threshold={thresh}\\nMetal: {metal_percent:.1f}%')
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
        
        print(f"Threshold {thresh}: {metal_percent:.1f}% metal pixels")
    
    plt.tight_layout()
    plt.savefig('threshold_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Threshold comparison saved as 'threshold_comparison.png'")
    plt.show()
    
    # Recommend best threshold (something reasonable, not too high or low)
    reasonable_thresholds = [r for r in results if 1 < r['metal_percent'] < 20]
    if reasonable_thresholds:
        best = min(reasonable_thresholds, key=lambda x: abs(x['metal_percent'] - 5))  # Target ~5%
        print(f"\\n✓ Recommended threshold: {best['threshold']} (gives {best['metal_percent']:.1f}% metal)")
        return best['threshold']
    else:
        print(f"\\n⚠️  All thresholds give extreme results. Using default 0.5")
        return 0.5

if __name__ == "__main__":
    mask_path = "./Polyner/input/mask_0.nii"
    
    # First test different thresholds
    recommended_threshold = test_different_thresholds(mask_path)
    
    # Apply the recommended threshold
    fixed_mask = fix_mask_to_binary(mask_path, threshold=recommended_threshold)