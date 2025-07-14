#!/usr/bin/env python3
"""
Prepare 2SOD symmetric geometry dataset for Polyner model training
Loads data from OneDrive 2SOD folder and converts to Polyner format
"""

import numpy as np
import SimpleITK as sitk
import os
from pathlib import Path
import json


def load_binary_data(file_path, shape, dtype=np.float32):
    """Load binary data and reshape."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = np.fromfile(str(file_path), dtype=dtype)
    return data.reshape(shape)


def save_as_nifti(data, output_path):
    """Save numpy array as NIFTI file."""
    if data.ndim == 2:
        data = data.astype(np.float32)
    img = sitk.GetImageFromArray(data)
    sitk.WriteImage(img, str(output_path))
    print(f"  Saved: {output_path}")
    return img


def generate_2SOD_fan_positions(num_detectors=400, SOD=410):
    """Generate fan sensor positions for symmetric 2SOD geometry."""
    # For symmetric geometry with SDD = 2*SOD, use arc detector
    # Detector spans from -gamma_max to +gamma_max
    gamma_max = np.arctan(108 / SOD)  # Approximate max fan angle
    
    # Generate detector angles
    detector_angles = np.linspace(-gamma_max, gamma_max, num_detectors)
    
    # Convert to positions (arc geometry)
    SDD = 2 * SOD  # 820 mm
    detector_positions = SDD * np.tan(detector_angles)
    
    return detector_positions.astype(np.float32)


def create_metal_mask(image, method='threshold', threshold_percentile=95):
    """Create metal mask from image."""
    if method == 'threshold':
        threshold = np.percentile(image, threshold_percentile)
        mask = (image > threshold).astype(np.float32)
    else:
        # Could add other segmentation methods here
        mask = np.zeros_like(image, dtype=np.float32)
    
    return mask


def prepare_2SOD_data_for_polyner():
    """Prepare 2SOD data for Polyner training."""
    
    # Paths
    data_base = Path("2SOD_data")  # Where OneDrive 2SOD folder is placed
    polyner_input = Path("input_2SOD")
    polyner_input.mkdir(exist_ok=True)
    
    print("Preparing 2SOD symmetric geometry data for Polyner...")
    print(f"Looking for data in: {data_base}")
    
    # 1. Load original slice (slice42)
    print("\n1. Loading original slice...")
    slice_files = [
        data_base / "slice42_216_216.bin",
        data_base / "slice42.bin"
    ]
    
    slice42 = None
    for slice_file in slice_files:
        if slice_file.exists():
            try:
                slice42 = load_binary_data(slice_file, (216, 216))
                print(f"  ✅ Loaded: {slice_file}")
                print(f"  Shape: {slice42.shape}, range: [{np.min(slice42):.4f}, {np.max(slice42):.4f}]")
                break
            except:
                continue
    
    if slice42 is None:
        print("  ❌ Could not load slice42 - check file exists and format")
        return False
    
    # Save as ground truth
    save_as_nifti(slice42, polyner_input / "gt_0.nii")
    
    # 2. Load sinogram (sino42)
    print("\n2. Loading sinogram...")
    sino_files = [
        data_base / "sino42_400_360.bin",
        data_base / "sino42.bin"
    ]
    
    sino42 = None
    for sino_file in sino_files:
        if sino_file.exists():
            try:
                # Try different shapes - sinogram could be (angles, detectors) or (detectors, angles)
                shapes_to_try = [(360, 400), (400, 360)]
                for shape in shapes_to_try:
                    try:
                        sino42 = load_binary_data(sino_file, shape)
                        print(f"  ✅ Loaded: {sino_file} with shape {shape}")
                        print(f"  Range: [{np.min(sino42):.4f}, {np.max(sino42):.4f}]")
                        break
                    except:
                        continue
                if sino42 is not None:
                    break
            except:
                continue
    
    if sino42 is None:
        print("  ❌ Could not load sino42 - check file exists and format")
        return False
    
    # Ensure sinogram is in (angles, detectors) format
    if sino42.shape[0] == 400 and sino42.shape[1] == 360:
        sino42 = sino42.T  # Transpose to (360, 400)
        print("  📝 Transposed sinogram to (angles, detectors) format")
    
    save_as_nifti(sino42, polyner_input / "ma_sinogram_0.nii")
    
    # 3. Load reconstruction for verification (rec42)
    print("\n3. Loading verification reconstruction...")
    rec_files = [
        data_base / "rec42_216_216.bin",
        data_base / "rec42.bin"
    ]
    
    rec42 = None
    for rec_file in rec_files:
        if rec_file.exists():
            try:
                rec42 = load_binary_data(rec_file, (216, 216))
                print(f"  ✅ Loaded: {rec_file}")
                print(f"  Shape: {rec42.shape}, range: [{np.min(rec42):.4f}, {np.max(rec42):.4f}]")
                break
            except:
                continue
    
    if rec42 is None:
        print("  ⚠️  Could not load rec42 - verification will be limited")
    else:
        save_as_nifti(rec42, polyner_input / "ma_0.nii")
    
    # 4. Generate fan sensor positions
    print("\n4. Generating fan sensor positions...")
    fan_positions = generate_2SOD_fan_positions(400, 410)
    fanSensorPos = fan_positions.reshape(-1, 1)
    fanPos_sitk = sitk.GetImageFromArray(fanSensorPos)
    sitk.WriteImage(fanPos_sitk, str(polyner_input / "fanSensorPos.nii"))
    
    print(f"  ✅ Generated fan positions:")
    print(f"  Shape: {fanSensorPos.shape}")
    print(f"  Range: {np.min(fan_positions):.1f} to {np.max(fan_positions):.1f} mm")
    print(f"  Detector spacing: {(np.max(fan_positions) - np.min(fan_positions))/399:.3f} mm")
    
    # 5. Create metal mask
    print("\n5. Creating metal mask...")
    metal_mask = create_metal_mask(slice42, method='threshold', threshold_percentile=95)
    save_as_nifti(metal_mask, polyner_input / "mask_0.nii")
    
    print(f"  ✅ Metal mask created:")
    print(f"  Metal pixels: {np.sum(metal_mask):.0f}")
    print(f"  Metal percentage: {100*np.sum(metal_mask)/metal_mask.size:.1f}%")
    
    # 6. Copy X-ray spectrum
    print("\n6. Setting up X-ray spectrum...")
    spectrum_files = [
        Path("input/GE14Spectrum120KVP.mat"),
        Path("input/spectrum_UNC.mat"),
        data_base / "GE14Spectrum120KVP.mat"
    ]
    
    spectrum_copied = False
    for spectrum_file in spectrum_files:
        if spectrum_file.exists():
            import shutil
            dest_path = polyner_input / spectrum_file.name
            shutil.copy2(spectrum_file, dest_path)
            print(f"  ✅ Copied: {spectrum_file.name}")
            spectrum_copied = True
            break
    
    if not spectrum_copied:
        print("  ⚠️  No spectrum file found - you may need to provide one")
    
    print(f"\n✅ 2SOD data preparation complete!")
    print(f"Output directory: {polyner_input}")
    print(f"Ready for Polyner training")
    
    return True


def create_2SOD_config():
    """Create configuration file for 2SOD symmetric geometry."""
    
    config = {
        "file": {
            "in_dir": "./input_2SOD",
            "model_dir": "./model_2SOD", 
            "out_dir": "./output_2SOD",
            "voxel_size": 1.0,
            "SOD": 410,           # Source-to-object distance
            "SDD": 820,           # Source-to-detector distance (2 × SOD)
            "detector_geometry": "arc",  # Use arc for symmetric geometry
            "geometry_type": "symmetric_fan_beam",
            "h": 216,
            "w": 216
        },
        "train": {
            "gpu": 0,
            "lr": 0.001,
            "epoch": 2000,
            "save_epoch": 500,
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
    
    with open("config_2SOD.json", "w") as f:
        json.dump(config, f, indent=4)
    
    print("✅ Created 2SOD configuration: config_2SOD.json")
    print(f"   SOD: {config['file']['SOD']} mm")
    print(f"   SDD: {config['file']['SDD']} mm (2 × SOD)")
    print(f"   Geometry: {config['file']['geometry_type']}")


if __name__ == "__main__":
    # Create necessary directories
    Path("model_2SOD").mkdir(exist_ok=True)
    Path("output_2SOD").mkdir(exist_ok=True)
    
    # Prepare data
    success = prepare_2SOD_data_for_polyner()
    
    if success:
        # Create configuration
        create_2SOD_config()
        
        print("\n🎯 Next steps:")
        print("1. Place 2SOD data files in ./2SOD_data/ directory:")
        print("   - slice42_216_216.bin (or slice42.bin)")
        print("   - sino42_400_360.bin (or sino42.bin)")
        print("   - rec42_216_216.bin (or rec42.bin)")
        print("2. Run: python main.py --config config_2SOD.json --img_id 0")
        print("3. Check results in output_2SOD/")
    else:
        print("\n❌ Data preparation failed - check file paths and formats")