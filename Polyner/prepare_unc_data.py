#!/usr/bin/env python3
"""
Prepare UNC test data for Polyner model training
Converts UNC RANDO phantom data to Polyner input format
"""

import numpy as np
import SimpleITK as sitk
import os
from pathlib import Path
import scipy.io as sio


def convert_hu_to_linear_attenuation(hu_data, mu_water=0.192):
    """Convert Hounsfield Units to linear attenuation coefficients."""
    return (hu_data / 1000.0) * mu_water + mu_water


def prepare_unc_data_for_polyner():
    """Prepare UNC data for Polyner training."""
    
    # Paths
    unc_base = Path("UNCtestdata")
    converted_path = unc_base / "converted"
    polyner_input = Path("input_unc")
    polyner_input.mkdir(exist_ok=True)
    
    print("Preparing UNC data for Polyner...")
    
    # 1. Load and process the 3D volume
    print("Processing 3D volume...")
    volume_file = converted_path / "RANDO_Metal_HU_480x480x120.nii"
    volume_sitk = sitk.ReadImage(str(volume_file))
    volume_hu = sitk.GetArrayFromImage(volume_sitk)  # Shape: (120, 480, 480)
    
    # Convert to linear attenuation coefficients
    volume_mu = convert_hu_to_linear_attenuation(volume_hu)
    
    # For 2D Polyner, we'll use middle slices as test cases
    mid_slice_idx = volume_hu.shape[0] // 2  # Around slice 60
    
    # Select a few slices around the middle where metal is prominent
    test_slices = [mid_slice_idx - 5, mid_slice_idx, mid_slice_idx + 5]
    
    # 2. Load projection data
    print("Processing projection data...")
    proj_file = unc_base / "Proj_RANDO_Metal_DEMSCBCT_src5_110kvp_744_229.bin"
    proj_data = np.fromfile(str(proj_file), dtype=np.float32)
    
    # Reshape to (angles, detector_rows, detector_cols) 
    proj_3d = proj_data.reshape(360, 744, 229)
    print(f"Projection data shape: {proj_3d.shape}")
    
    # For 2D reconstruction, we need to extract relevant projections
    # corresponding to our selected slices
    
    # 3. Create Polyner input files for each test slice
    for i, slice_idx in enumerate(test_slices):
        print(f"\nProcessing slice {slice_idx} (case {i})...")
        
        # Ground truth (convert from HU to linear attenuation, then resize to 256x256)
        gt_slice = volume_mu[slice_idx]  # Shape: (480, 480)
        
        # Resize to match segmentation size (216x216)
        gt_sitk = sitk.GetImageFromArray(gt_slice)
        gt_resized = sitk.Resample(gt_sitk, (216, 216), sitk.Transform(), 
                                  sitk.sitkLinear, gt_sitk.GetOrigin(), 
                                  (gt_sitk.GetSpacing()[0] * 480/216, 
                                   gt_sitk.GetSpacing()[1] * 480/216),
                                  gt_sitk.GetDirection(), 0.0, gt_sitk.GetPixelID())
        gt_array = sitk.GetArrayFromImage(gt_resized)
        
        # Save ground truth
        gt_output = sitk.GetImageFromArray(gt_array.astype(np.float32))
        sitk.WriteImage(gt_output, str(polyner_input / f"gt_{i}.nii"))
        
        # Metal mask (using our generated mask, extract same slice)
        mask_file = converted_path / "metal_mask_RANDO_permissive.nii"
        if mask_file.exists():
            mask_sitk = sitk.ReadImage(str(mask_file))
            mask_volume = sitk.GetArrayFromImage(mask_sitk)
            mask_slice = mask_volume[slice_idx].astype(np.float32)
            
            # Resize mask to 216x216
            mask_slice_sitk = sitk.GetImageFromArray(mask_slice)
            mask_resized = sitk.Resample(mask_slice_sitk, (216, 216), sitk.Transform(),
                                       sitk.sitkNearestNeighbor)
            mask_array = sitk.GetArrayFromImage(mask_resized)
            
            # Save mask
            mask_output = sitk.GetImageFromArray(mask_array.astype(np.float32))
            sitk.WriteImage(mask_output, str(polyner_input / f"mask_{i}.nii"))
        
        # For projections, we need to extract the relevant detector rows
        # that correspond to this slice. For cone beam, each slice
        # contributes to multiple projections depending on cone angle
        
        # Estimate which detector rows correspond to this slice
        # Assuming central ray alignment and uniform spacing
        total_slices = volume_hu.shape[0]  # 120
        detector_rows = proj_3d.shape[1]   # 744
        
        # Map slice index to detector row range
        rows_per_slice = detector_rows / total_slices
        start_row = int(slice_idx * rows_per_slice)
        end_row = int((slice_idx + 1) * rows_per_slice)
        
        # Extract projections for this slice (average the relevant rows)
        slice_projections = np.mean(proj_3d[:, start_row:end_row, :], axis=1)  # Shape: (360, 229)
        
        # Resize to match expected sinogram size for 216x216 reconstruction
        # Adjust detector count to match smaller image size
        from scipy import ndimage
        target_detectors = 216  # Match image dimensions
        
        # Interpolate along detector dimension
        sinogram_resized = np.zeros((360, target_detectors))
        for angle in range(360):
            sinogram_resized[angle, :] = np.interp(
                np.linspace(0, 229-1, target_detectors),
                np.arange(229),
                slice_projections[angle, :]
            )
        
        # Convert to negative log (if needed - depends on data format)
        # The projection data seems to already be in sinogram format
        ma_sinogram = sinogram_resized.astype(np.float32)
        
        # Save sinogram
        sinogram_sitk = sitk.GetImageFromArray(ma_sinogram)
        sitk.WriteImage(sinogram_sitk, str(polyner_input / f"ma_sinogram_{i}.nii"))
        
        print(f"  GT shape: {gt_array.shape}, range: [{np.min(gt_array):.4f}, {np.max(gt_array):.4f}]")
        print(f"  Sinogram shape: {ma_sinogram.shape}, range: [{np.min(ma_sinogram):.4f}, {np.max(ma_sinogram):.4f}]")
        if mask_file.exists():
            print(f"  Mask shape: {mask_array.shape}, metal pixels: {np.sum(mask_array)}")
    
    # 4. Create fan sensor positions (linear detector geometry for UNC cone beam)
    print("\nCreating detector geometry...")
    
    # UNC linear detector parameters
    detector_width = 148.8  # mm
    detector_pixels = 216   # resized from 744
    detector_pixel_size = detector_width / detector_pixels
    detector_offset = 70.5  # mm
    
    # Linear detector positions (not arc) - centered with offset
    detector_positions = np.linspace(-detector_width/2 + detector_offset, 
                                   detector_width/2 + detector_offset, 
                                   detector_pixels).astype(np.float32)
    fanSensorPos = detector_positions.reshape(-1, 1)
    
    fanPos_sitk = sitk.GetImageFromArray(fanSensorPos)
    sitk.WriteImage(fanPos_sitk, str(polyner_input / "fanSensorPos.nii"))
    
    # 5. Copy X-ray spectrum
    spectrum_source = Path("input/GE14Spectrum120KVP.mat")
    spectrum_dest = polyner_input / "GE14Spectrum120KVP.mat"
    if spectrum_source.exists():
        import shutil
        shutil.copy2(spectrum_source, spectrum_dest)
        print("Copied X-ray spectrum file")
    
    # 6. Create metal-affected reconstructions (placeholder)
    print("Creating metal-affected reconstructions...")
    for i in range(len(test_slices)):
        # For now, use the original HU data converted to linear attenuation
        slice_idx = test_slices[i]
        ma_slice = volume_mu[slice_idx]
        
        # Resize to 216x216
        ma_sitk = sitk.GetImageFromArray(ma_slice)
        ma_resized = sitk.Resample(ma_sitk, (216, 216), sitk.Transform(),
                                  sitk.sitkLinear)
        ma_array = sitk.GetArrayFromImage(ma_resized)
        
        ma_output = sitk.GetImageFromArray(ma_array.astype(np.float32))
        sitk.WriteImage(ma_output, str(polyner_input / f"ma_{i}.nii"))
    
    print(f"\nUNC data preparation complete!")
    print(f"Output directory: {polyner_input}")
    print(f"Created {len(test_slices)} test cases")
    
    return len(test_slices)


def create_unc_config():
    """Create configuration file adapted for UNC data."""
    
    # Based on UNC geometry and our data preparation
    config = {
        "file": {
            "in_dir": "./input_unc",
            "model_dir": "./model_unc", 
            "out_dir": "./output_unc",
            "voxel_size": 0.2,  # UNC detector pixel size
            "SOD": 410,         # UNC source-to-object distance
            "h": 216,
            "w": 216
        },
        "train": {
            "gpu": 0,
            "lr": 1e-3,
            "epoch": 2000,      # Reduced for testing
            "save_epoch": 1000,
            "num_sample_ray": 2,
            "lr_decay_epoch": 1000,
            "lr_decay_coefficient": 0.1,
            "batch_size": 40,
            "lambda": 0.2
        },
        "encoding": {
            "otype": "Grid",
            "type": "Hash", 
            "n_levels": 16,
            "n_features_per_level": 8,
            "log2_hashmap_size": 19,
            "base_resolution": 2,
            "per_level_scale": 2,
            "interpolation": "Linear"
        },
        "network": {
            "otype": "FullyFusedMLP",
            "activation": "ReLU", 
            "output_activation": "Squareplus",
            "n_neurons": 128,
            "n_hidden_layers": 2
        }
    }
    
    import json
    with open("config_unc.json", "w") as f:
        json.dump(config, f, indent=4)
    
    print("Created UNC configuration: config_unc.json")


if __name__ == "__main__":
    num_cases = prepare_unc_data_for_polyner()
    create_unc_config()
    
    print("\nNext steps:")
    print("1. Review the prepared data in input_unc/")
    print("2. Run: python main.py --config config_unc.json")
    print(f"3. Check results for {num_cases} test cases")