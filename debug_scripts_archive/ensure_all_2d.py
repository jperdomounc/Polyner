import SimpleITK as sitk
import nibabel as nib
import numpy as np
import os

def ensure_2d_format(input_dir):
    """Ensure all image files in input directory are 2D format for Polyner"""
    
    files_to_check = ['mask_0.nii', 'ma_0.nii', 'gt_0.nii', 'ma_0']
    
    for filename in files_to_check:
        filepath = os.path.join(input_dir, filename)
        
        if os.path.exists(filepath):
            print(f"Checking {filename}...")
            
            try:
                # Load with SimpleITK (as Polyner does)
                sitk_img = sitk.ReadImage(filepath)
                data = sitk.GetArrayFromImage(sitk_img)
                
                print(f"  Current shape: {data.shape}")
                print(f"  Current dtype: {data.dtype}")
                
                # Check if it's 3D with last dimension = 1
                if len(data.shape) == 3 and data.shape[2] == 1:
                    print(f"  Converting {filename} from 3D to 2D...")
                    
                    # Squeeze to 2D
                    data_2d = data.squeeze(axis=2)
                    
                    # Create new 2D SimpleITK image
                    sitk_2d = sitk.GetImageFromArray(data_2d)
                    sitk_2d.SetSpacing(sitk_img.GetSpacing()[:2])  # Keep only 2D spacing
                    sitk_2d.SetOrigin(sitk_img.GetOrigin()[:2])    # Keep only 2D origin
                    
                    # Save the 2D version
                    sitk.WriteImage(sitk_2d, filepath)
                    print(f"  ✓ Converted {filename} to 2D shape: {data_2d.shape}")
                    
                elif len(data.shape) == 2:
                    print(f"  ✓ {filename} is already 2D")
                    
                else:
                    print(f"  ⚠️  Unexpected shape for {filename}: {data.shape}")
                    
            except Exception as e:
                print(f"  ❌ Error processing {filename}: {e}")
        else:
            print(f"  - {filename} not found (skipping)")
    
    print("\nVerification - checking all files again:")
    for filename in files_to_check:
        filepath = os.path.join(input_dir, filename)
        if os.path.exists(filepath):
            try:
                data = sitk.GetArrayFromImage(sitk.ReadImage(filepath))
                print(f"  {filename}: shape {data.shape} ✓")
            except Exception as e:
                print(f"  {filename}: Error - {e}")

if __name__ == "__main__":
    input_directory = "/Users/juanperdomo/Desktop/input_backup"
    ensure_2d_format(input_directory)