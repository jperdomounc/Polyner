import SimpleITK as sitk
import nibabel as nib
import numpy as np
import os

def compare_file_formats(input_dir, compare_dir):
    """Compare file formats between input and input_compare directories"""
    
    print("=== DATA FORMAT COMPARISON ===\n")
    
    # Map files with their counterparts
    file_mapping = {
        'mask_0.nii': 'mask_0 (1).nii',
        'ma_0.nii': 'ma_0 (1).nii',
        'gt_0.nii': 'gt_0 (1).nii',
        'ma_sinogram_0.nii': 'ma_sinogram_0 (2).nii',
        'fanSensorPos.nii': 'fanSensorPos.nii',
        'GE14Spectrum120KVP.mat': 'GE14Spectrum120KVP (1).mat'
    }
    
    for input_file, compare_file in file_mapping.items():
        input_path = os.path.join(input_dir, input_file)
        compare_path = os.path.join(compare_dir, compare_file)
        
        print(f"--- {input_file} vs {compare_file} ---")
        
        if os.path.exists(input_path) and os.path.exists(compare_path):
            try:
                if input_file.endswith('.nii'):
                    # Load with SimpleITK (as Polyner does)
                    input_data = sitk.GetArrayFromImage(sitk.ReadImage(input_path))
                    compare_data = sitk.GetArrayFromImage(sitk.ReadImage(compare_path))
                    
                    print(f"INPUT:   shape={input_data.shape}, dtype={input_data.dtype}")
                    print(f"         min={input_data.min():.6f}, max={input_data.max():.6f}")
                    print(f"         mean={input_data.mean():.6f}, std={input_data.std():.6f}")
                    
                    print(f"COMPARE: shape={compare_data.shape}, dtype={compare_data.dtype}")
                    print(f"         min={compare_data.min():.6f}, max={compare_data.max():.6f}")
                    print(f"         mean={compare_data.mean():.6f}, std={compare_data.std():.6f}")
                    
                    # Check if shapes match
                    if input_data.shape != compare_data.shape:
                        print(f"⚠️  SHAPE MISMATCH! {input_data.shape} vs {compare_data.shape}")
                    else:
                        print("✓ Shapes match")
                    
                    # Check if data ranges are similar
                    if abs(input_data.min() - compare_data.min()) > 0.01 or abs(input_data.max() - compare_data.max()) > 0.01:
                        print(f"⚠️  DATA RANGE DIFFERENCE!")
                    else:
                        print("✓ Data ranges similar")
                        
                    # Check unique values for binary masks
                    if 'mask' in input_file.lower():
                        input_unique = np.unique(input_data)
                        compare_unique = np.unique(compare_data)
                        print(f"INPUT unique values:   {input_unique}")
                        print(f"COMPARE unique values: {compare_unique}")
                        
                elif input_file.endswith('.mat'):
                    print("MAT file - checking file sizes...")
                    input_size = os.path.getsize(input_path)
                    compare_size = os.path.getsize(compare_path)
                    print(f"INPUT size:   {input_size} bytes")
                    print(f"COMPARE size: {compare_size} bytes")
                    if input_size != compare_size:
                        print("⚠️  FILE SIZE DIFFERENCE!")
                    else:
                        print("✓ File sizes match")
                        
            except Exception as e:
                print(f"❌ Error comparing {input_file}: {e}")
        else:
            print(f"❌ Missing files: input={os.path.exists(input_path)}, compare={os.path.exists(compare_path)}")
        
        print()

if __name__ == "__main__":
    input_dir = "/Users/juanperdomo/Desktop/PolynerCode/Polyner/input"
    compare_dir = "/Users/juanperdomo/Desktop/PolynerCode/input_compare"
    
    compare_file_formats(input_dir, compare_dir)