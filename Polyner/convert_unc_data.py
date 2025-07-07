#!/usr/bin/env python3
"""
UNC Test Data Converter
Converts binary UNC test data to NIFTI format for processing with metal_mask_threshold.py
"""

import numpy as np
import SimpleITK as sitk
import os
from pathlib import Path
import argparse


def read_binary_volume(filename, shape, dtype=np.float32):
    """
    Read binary volume data.
    
    Args:
        filename (str): Path to binary file
        shape (tuple): Dimensions (width, height, depth) or (width, height)
        dtype: Data type for reading
        
    Returns:
        np.ndarray: Volume data
    """
    try:
        data = np.fromfile(filename, dtype=dtype)
        expected_size = np.prod(shape)
        
        if len(data) != expected_size:
            print(f"Warning: File size mismatch. Expected {expected_size}, got {len(data)}")
            # Try different data types
            for test_dtype in [np.float64, np.int32, np.int16, np.uint16]:
                test_data = np.fromfile(filename, dtype=test_dtype)
                if len(test_data) == expected_size:
                    print(f"Using dtype {test_dtype}")
                    data = test_data.astype(dtype)
                    break
            else:
                # Truncate or pad as needed
                if len(data) > expected_size:
                    data = data[:expected_size]
                else:
                    data = np.pad(data, (0, expected_size - len(data)))
        
        return data.reshape(shape)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None


def convert_unc_files():
    """Convert UNC binary files to NIFTI format."""
    base_path = Path("/Users/juanperdomo/Desktop/PolynerCode/Polyner/UNCtestdata")
    output_path = base_path / "converted"
    output_path.mkdir(exist_ok=True)
    
    # File specifications based on filenames
    files_to_convert = [
        {
            "filename": "Rec_RANDO_Metal_DEMSCBCT_src5_110kvp_480_480_120_75keV_HU.bin",
            "shape": (480, 480, 120),
            "output_name": "RANDO_Metal_HU_480x480x120.nii",
            "dtype": np.float32
        },
        {
            "filename": "Rec_RANDO_Metal_DEMSCBCT_src5_110kvp_480_480_120_75keV_mu.bin", 
            "shape": (480, 480, 120),
            "output_name": "RANDO_Metal_mu_480x480x120.nii",
            "dtype": np.float32
        },
        {
            "filename": "rec42_216_216.bin",
            "shape": (216, 216),
            "output_name": "rec42_216x216.nii",
            "dtype": np.float32
        },
        {
            "filename": "slice42_216_216.bin",
            "shape": (216, 216),
            "output_name": "slice42_216x216.nii",
            "dtype": np.float32
        }
    ]
    
    converted_files = []
    
    for file_info in files_to_convert:
        input_file = base_path / file_info["filename"]
        output_file = output_path / file_info["output_name"]
        
        if not input_file.exists():
            print(f"File not found: {input_file}")
            continue
            
        print(f"Converting {input_file.name}...")
        
        # Read binary data
        data = read_binary_volume(str(input_file), file_info["shape"], file_info["dtype"])
        
        if data is None:
            continue
            
        # Convert to SimpleITK image
        # For 3D data, we need to transpose to match ITK convention (z, y, x)
        if len(data.shape) == 3:
            data = np.transpose(data, (2, 1, 0))  # (x, y, z) -> (z, y, x)
        
        image = sitk.GetImageFromArray(data)
        
        # Set basic spacing (can be adjusted based on actual scanner parameters)
        if len(data.shape) == 3:
            image.SetSpacing([1.0, 1.0, 1.0])  # mm
        else:
            image.SetSpacing([1.0, 1.0])  # mm
            
        # Write NIFTI file
        sitk.WriteImage(image, str(output_file))
        converted_files.append(str(output_file))
        
        print(f"  -> Saved as {output_file.name}")
        print(f"  -> Shape: {data.shape}, Range: [{np.min(data):.2f}, {np.max(data):.2f}]")
    
    return converted_files


def convert_vmi_data():
    """Convert VMI HU data for testing."""
    base_path = Path("/Users/juanperdomo/Desktop/PolynerCode/Polyner/UNCtestdata")
    vmi_path = base_path / "DualEnergy Result" / "DEMSCBCT" / "RANDO_MAR_Ca_110kVp_noconstrain" / "VMI HU"
    output_path = base_path / "converted" / "VMI_HU"
    output_path.mkdir(parents=True, exist_ok=True)
    
    converted_files = []
    
    # Process a few VMI energy levels
    test_energies = [70, 75, 80, 100, 120]  # keV
    
    for energy in test_energies:
        vmi_file = vmi_path / f"RANDO_MAR_Ca_110kVp_VMI_HU_{energy}_keV.bin"
        
        if not vmi_file.exists():
            continue
            
        # Estimate dimensions based on file size
        file_size = vmi_file.stat().st_size
        
        # Try common CT dimensions
        possible_shapes =  (216, 216) #using single slice
        
        for shape in possible_shapes:
            if file_size == np.prod(shape) * 4:  # 4 bytes for float32
                print(f"Converting VMI {energy} keV data...")
                
                data = read_binary_volume(str(vmi_file), shape, np.float32)
                
                if data is not None:
                    if len(data.shape) == 3:
                        data = np.transpose(data, (2, 1, 0))
                    
                    image = sitk.GetImageFromArray(data)
                    
                    if len(data.shape) == 3:
                        image.SetSpacing([1.0, 1.0, 1.0])
                    else:
                        image.SetSpacing([1.0, 1.0])
                    
                    output_file = output_path / f"VMI_HU_{energy}keV.nii"
                    sitk.WriteImage(image, str(output_file))
                    converted_files.append(str(output_file))
                    
                    print(f"  -> {output_file.name}: {data.shape}, Range: [{np.min(data):.2f}, {np.max(data):.2f}]")
                    break
    
    return converted_files


def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(description="Convert UNC binary data to NIFTI")
    parser.add_argument("--vmi", action="store_true", help="Also convert VMI data")
    args = parser.parse_args()
    
    print("Converting UNC test data to NIFTI format...")
    
    # Convert main files
    converted_files = convert_unc_files()
    
    # Convert VMI files if requested
    if args.vmi:
        print("\nConverting VMI data...")
        vmi_files = convert_vmi_data()
        converted_files.extend(vmi_files)
    
    print(f"\nConversion complete! {len(converted_files)} files converted.")
    
    return converted_files


if __name__ == "__main__":
    main()