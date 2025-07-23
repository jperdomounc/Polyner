import SimpleITK as sitk
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_and_normalize_data():
    """Analyze and normalize NIfTI data to match original repository ranges"""
    
    print("=== NIFTI NORMALIZATION ANALYSIS ===\n")
    
    input_dir = './Polyner/input'
    compare_dir = './input_compare'
    
    # File mappings
    files = {
        'ma_0.nii': 'ma_0 (1).nii',
        'gt_0.nii': 'gt_0 (1).nii', 
        'mask_0.nii': 'mask_0 (1).nii'
    }
    
    results = {}
    
    for current_file, original_file in files.items():
        print(f"--- Processing {current_file} ---")
        
        # Load current data
        current_path = f'{input_dir}/{current_file}'
        current_img = sitk.ReadImage(current_path)
        current_data = sitk.GetArrayFromImage(current_img)
        
        # Load original for comparison
        try:
            original_data = sitk.GetArrayFromImage(sitk.ReadImage(f'{compare_dir}/{original_file}'))
            has_original = True
        except:
            original_data = None
            has_original = False
            
        print(f"Current data  : range=[{current_data.min():8.3f}, {current_data.max():8.3f}], mean={current_data.mean():7.3f}")
        
        if has_original:
            print(f"Original data : range=[{original_data.min():8.3f}, {original_data.max():8.3f}], mean={original_data.mean():7.3f}")
            
            # Determine normalization strategy
            if 'mask' in current_file:
                # Mask should already be binary - no normalization needed
                normalized_data = current_data.copy()
                strategy = "No normalization (already binary)"
                
            elif 'gt' in current_file:
                # Ground truth: normalize to match original range [0, 0.418]
                # This represents attenuation coefficients - must be physical values
                current_min, current_max = current_data.min(), current_data.max()
                target_min, target_max = 0.0, 0.418
                
                # Min-max normalization to target range
                normalized_data = (current_data - current_min) / (current_max - current_min) * (target_max - target_min) + target_min
                
                # Ensure no negative values for attenuation coefficients
                normalized_data = np.maximum(normalized_data, 0.0)
                
                strategy = f"Min-max normalize to [{target_min}, {target_max}] (attenuation coefficients)"
                
            elif 'ma' in current_file:
                # Metal artifact image: normalize to reasonable attenuation range
                # Allow some negative values for preprocessing artifacts, but keep physical
                current_min, current_max = current_data.min(), current_data.max()
                target_min, target_max = -0.2, 2.5  # Reasonable attenuation range
                
                # Min-max normalization
                normalized_data = (current_data - current_min) / (current_max - current_min) * (target_max - target_min) + target_min
                
                strategy = f"Min-max normalize to [{target_min}, {target_max}] (metal artifact range)"
                
            else:
                normalized_data = current_data.copy()
                strategy = "No normalization applied"
                
            print(f"Strategy      : {strategy}")
            print(f"Normalized    : range=[{normalized_data.min():8.3f}, {normalized_data.max():8.3f}], mean={normalized_data.mean():7.3f}")
            
            # Calculate normalization impact
            range_ratio = (normalized_data.max() - normalized_data.min()) / (current_data.max() - current_data.min())
            print(f"Range ratio   : {range_ratio:.3f} (normalized/original)")
            
            results[current_file] = {
                'original_data': current_data,
                'normalized_data': normalized_data,
                'strategy': strategy,
                'original_range': (current_data.min(), current_data.max()),
                'normalized_range': (normalized_data.min(), normalized_data.max()),
                'sitk_img': current_img,
                'path': current_path
            }
            
        else:
            print("Original data : Not available")
            results[current_file] = {'original_data': current_data, 'normalized_data': current_data.copy()}
            
        print()
    
    return results