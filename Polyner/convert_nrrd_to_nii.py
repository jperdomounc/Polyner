#!/usr/bin/env python3
"""
NRRD to NIFTI Converter
Converts segmentation NRRD files to NIFTI format for processing
"""

import SimpleITK as sitk
import numpy as np
import argparse
from pathlib import Path


def convert_nrrd_to_nii(nrrd_file, output_file=None, verbose=True):
    """
    Convert NRRD file to NIFTI format.
    
    Args:
        nrrd_file (str): Path to input NRRD file
        output_file (str, optional): Path to output NII file
        verbose (bool): Print conversion details
        
    Returns:
        str: Path to output file
    """
    nrrd_path = Path(nrrd_file)
    
    if not nrrd_path.exists():
        raise FileNotFoundError(f"NRRD file not found: {nrrd_file}")
    
    # Generate output filename if not provided
    if output_file is None:
        output_file = nrrd_path.with_suffix('.nii').name
    
    output_path = Path(output_file)
    
    if verbose:
        print(f"Converting: {nrrd_path.name}")
        print(f"Output: {output_path}")
    
    try:
        # Read NRRD file
        image = sitk.ReadImage(str(nrrd_path))
        
        # Get image information
        array = sitk.GetArrayFromImage(image)
        
        if verbose:
            print(f"Image shape: {array.shape}")
            print(f"Data type: {array.dtype}")
            print(f"Spacing: {image.GetSpacing()}")
            print(f"Origin: {image.GetOrigin()}")
            print(f"Direction: {image.GetDirection()}")
            
            # For segmentation, show label statistics
            unique_values = np.unique(array)
            print(f"Unique labels: {unique_values}")
            for val in unique_values:
                count = np.sum(array == val)
                percentage = 100 * count / array.size
                print(f"  Label {val}: {count} voxels ({percentage:.2f}%)")
        
        # Write as NIFTI
        sitk.WriteImage(image, str(output_path))
        
        if verbose:
            print(f"✓ Conversion successful: {output_path}")
            
        return str(output_path)
        
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        raise


def batch_convert_nrrd(input_dir, output_dir=None, pattern="*.nrrd"):
    """
    Convert multiple NRRD files in a directory.
    
    Args:
        input_dir (str): Directory containing NRRD files
        output_dir (str, optional): Output directory for NII files
        pattern (str): File pattern to match
    """
    input_path = Path(input_dir)
    
    if output_dir is None:
        output_dir = input_path / "converted_nii"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    nrrd_files = list(input_path.glob(pattern))
    
    if not nrrd_files:
        print(f"No NRRD files found matching pattern '{pattern}' in {input_dir}")
        return
    
    print(f"Found {len(nrrd_files)} NRRD files to convert")
    print(f"Output directory: {output_path}")
    
    converted_files = []
    
    for nrrd_file in nrrd_files:
        try:
            output_file = output_path / nrrd_file.with_suffix('.nii').name
            converted_file = convert_nrrd_to_nii(nrrd_file, output_file)
            converted_files.append(converted_file)
            print()  # Add space between conversions
        except Exception as e:
            print(f"Failed to convert {nrrd_file.name}: {e}")
            print()
    
    print(f"Batch conversion complete: {len(converted_files)}/{len(nrrd_files)} files converted")
    return converted_files


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Convert NRRD segmentation files to NIFTI format"
    )
    
    parser.add_argument("input", help="Input NRRD file or directory")
    parser.add_argument("-o", "--output", help="Output NII file or directory")
    parser.add_argument("--batch", action="store_true", 
                       help="Batch convert directory of NRRD files")
    parser.add_argument("--pattern", default="*.nrrd",
                       help="File pattern for batch conversion (default: *.nrrd)")
    parser.add_argument("-q", "--quiet", action="store_true",
                       help="Suppress verbose output")
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch conversion
        converted_files = batch_convert_nrrd(
            args.input, 
            args.output, 
            args.pattern
        )
    else:
        # Single file conversion
        converted_file = convert_nrrd_to_nii(
            args.input, 
            args.output, 
            verbose=not args.quiet
        )
        print(f"Converted file: {converted_file}")


if __name__ == "__main__":
    # If run directly, convert the segmentation file we found
    segmentation_files = [
        "/Users/juanperdomo/Desktop/PolynerCode/Segmentation-Segment_1-label.nrrd",
        "/Users/juanperdomo/Desktop/PolynerCode/Polyner/UNCtestdata/Segmentation-Segment_1-label.nrrd"
    ]
    
    print("NRRD to NIFTI Converter")
    print("=" * 50)
    
    for seg_file in segmentation_files:
        if Path(seg_file).exists():
            print(f"\nConverting: {Path(seg_file).name}")
            try:
                output_name = f"UNCtestdata/converted/{Path(seg_file).stem}.nii"
                convert_nrrd_to_nii(seg_file, output_name)
            except Exception as e:
                print(f"Error: {e}")
        else:
            print(f"File not found: {seg_file}")
    
    print(f"\n{'='*50}")
    print("Conversion complete!")