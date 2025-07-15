#!/usr/bin/env python3
"""
Prepare 2SOD data for regular input/ folder
This will convert 2SOD data to work with the standard main.py pipeline
"""

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
import json
import shutil

def load_binary_data(file_path, shape, dtype=np.float32):
    """Load binary data and reshape."""
    print(f"  Loading: {file_path} -> {shape}")
    data = np.fromfile(str(file_path), dtype=dtype)
    return data.reshape(shape)

def save_as_nifti(data, output_path):
    """Save numpy array as NIFTI file."""
    if data.ndim == 2:
        data = data.astype(np.float32)
    img = sitk.GetImageFromArray(data)
    sitk.WriteImage(img, str(output_path))
    print(f"  ✅ Saved: {output_path}")

def generate_2SOD_fan_positions(num_detectors=400, SOD=410):
    """Generate fan sensor positions for symmetric 2SOD geometry."""
    gamma_max = np.arctan(108 / SOD)
    detector_angles = np.linspace(-gamma_max, gamma_max, num_detectors)
    SDD = 2 * SOD  # 820 mm
    detector_positions = SDD * np.tan(detector_angles)
    return detector_positions.astype(np.float32)

def find_2SOD_directory():
    """Find the 2SOD data directory."""
    possible_paths = [
        Path("UNCtestdata/2SOD"),
        Path("UNCtestdata/2sod"), 
        Path("unctestdata/2SOD"),
        Path("unctestdata/2sod")
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    return None

def prepare_2SOD_for_input():
    """Convert 2SOD data to standard input/ format."""
    print("🎯 Preparing 2SOD data for input/ folder...")
    
    # Find 2SOD directory
    data_base = find_2SOD_directory()
    if data_base is None:
        print("❌ Could not find 2SOD data directory")
        return False
    
    print(f"✅ Found data directory: {data_base}")
    
    # Show what files are available
    print("Files in directory:")
    for item in data_base.iterdir():
        if item.is_file():
            size_mb = item.stat().st_size / (1024*1024)
            print(f"  📄 {item.name} ({size_mb:.1f} MB)")
    
    # Load the three main files
    slice42 = None
    sino42 = None
    rec42 = None
    
    # Load slice42_2SOD.bin
    slice_file = data_base / "slice42_2SOD.bin"
    if slice_file.exists():
        print("\n📂 Loading slice42_2SOD.bin...")
        try:
            slice42 = load_binary_data(slice_file, (216, 216), np.float64)
            print(f"  Range: [{np.min(slice42):.3f}, {np.max(slice42):.3f}]")
        except:
            try:
                slice42 = load_binary_data(slice_file, (216, 216), np.float32)
                print(f"  Range: [{np.min(slice42):.3f}, {np.max(slice42):.3f}]")
            except Exception as e:
                print(f"  ❌ Failed: {e}")
    
    # Load sino42_2SOD.bin
    sino_file = data_base / "sino42_2SOD.bin"
    if sino_file.exists():
        print("\n📂 Loading sino42_2SOD.bin...")
        file_size = sino_file.stat().st_size
        print(f"  File size: {file_size} bytes")
        
        # Calculate possible shapes based on actual file size
        total_float32 = file_size // 4  # 187,200 elements
        total_float64 = file_size // 8  # 93,600 elements
        
        print(f"  Total elements: {total_float32} (float32) or {total_float64} (float64)")
        
        # Find factors that make sense for sinogram dimensions
        possible_shapes = []
        
        # For float32 (187,200 elements)
        for h in [360, 400, 744, 520, 468]:
            if total_float32 % h == 0:
                w = total_float32 // h
                possible_shapes.append(((h, w), np.float32))
                
        # For float64 (93,600 elements) 
        for h in [360, 400, 520, 468, 306]:
            if total_float64 % h == 0:
                w = total_float64 // h
                possible_shapes.append(((h, w), np.float64))
        
        print(f"  Possible shapes: {[(shape, dtype.__name__) for shape, dtype in possible_shapes]}")
        
        # Try all possible shapes
        for (shape, dtype) in possible_shapes:
            try:
                print(f"  Trying {shape} {dtype.__name__}...")
                sino42 = load_binary_data(sino_file, shape, dtype)
                
                # Check if this looks like reasonable sinogram data
                if np.min(sino42) >= 0 and np.max(sino42) > 0:
                    # Ensure (angles, detectors) format - typically angles < detectors
                    if shape[0] > shape[1]:
                        print(f"    Transposing from {sino42.shape} to {sino42.T.shape}")
                        sino42 = sino42.T
                    print(f"  ✅ Success: {sino42.shape}, range: [{np.min(sino42):.3f}, {np.max(sino42):.3f}]")
                    break
                else:
                    print(f"    Data range looks suspicious: [{np.min(sino42):.3f}, {np.max(sino42):.3f}]")
                    
            except Exception as e:
                print(f"    Failed: {e}")
                continue
    
    # Load rec42_2SOD.bin
    rec_file = data_base / "rec42_2SOD.bin"
    if rec_file.exists():
        print("\n📂 Loading rec42_2SOD.bin...")
        try:
            rec42 = load_binary_data(rec_file, (216, 216), np.float64)
            print(f"  Range: [{np.min(rec42):.3f}, {np.max(rec42):.3f}]")
        except:
            try:
                rec42 = load_binary_data(rec_file, (216, 216), np.float32)
                print(f"  Range: [{np.min(rec42):.3f}, {np.max(rec42):.3f}]")
            except Exception as e:
                print(f"  ❌ Failed: {e}")
    
    # Check what we got
    print(f"\n📋 Loading results:")
    print(f"  slice42: {'✅' if slice42 is not None else '❌'}")
    print(f"  sino42:  {'✅' if sino42 is not None else '❌'}")  
    print(f"  rec42:   {'✅' if rec42 is not None else '❌'}")
    
    if slice42 is None or sino42 is None:
        print("❌ Cannot proceed without slice42 and sino42")
        return False
    
    # Now save to input/ folder in Polyner format
    print(f"\n💾 Saving to input/ folder...")
    
    # Save as single test case (case 0)
    print(f"\nCreating test case 0...")
    
    # Ground truth
    save_as_nifti(slice42, f'input/gt_0.nii')
    
    # Sinogram
    save_as_nifti(sino42, f'input/ma_sinogram_0.nii')
    
    # Metal mask (create simple threshold mask)
    threshold = np.percentile(slice42, 95)
    metal_mask = (slice42 > threshold).astype(np.float32)
    save_as_nifti(metal_mask, f'input/mask_0.nii')
    print(f"  Metal pixels: {np.sum(metal_mask):.0f}")
    
    # Metal-affected reconstruction (use rec42 if available, else slice42)
    ma_image = rec42 if rec42 is not None else slice42
    save_as_nifti(ma_image, f'input/ma_0.nii')
    
    # Generate and save fan sensor positions for 2SOD geometry
    print(f"\n🔧 Creating 2SOD fan sensor positions...")
    
    # Use the actual sinogram detector count
    if sino42 is not None:
        num_detectors = sino42.shape[1]  # detectors dimension
        print(f"  Sinogram shape: {sino42.shape} -> using {num_detectors} detectors")
    else:
        num_detectors = 400  # fallback
        print(f"  Using default {num_detectors} detectors")
    
    fan_positions = generate_2SOD_fan_positions(num_detectors, 410)
    fanSensorPos = fan_positions.reshape(-1, 1)
    fanPos_sitk = sitk.GetImageFromArray(fanSensorPos)
    sitk.WriteImage(fanPos_sitk, 'input/fanSensorPos.nii')
    print(f"  ✅ Fan positions: {len(fan_positions)} detectors, range [{np.min(fan_positions):.1f}, {np.max(fan_positions):.1f}] mm")
    
    # Copy spectrum file if available
    spectrum_sources = [
        'UNCtestdata/spectrum_UNC.mat',
        'input/GE14Spectrum120KVP.mat'
    ]
    
    for spectrum_file in spectrum_sources:
        if Path(spectrum_file).exists():
            if 'spectrum_UNC' in spectrum_file:
                shutil.copy2(spectrum_file, 'input/spectrum_UNC.mat')
                print(f"  ✅ Copied UNC spectrum")
            break
    
    # Update main config.json for 2SOD geometry
    print(f"\n⚙️ Updating config.json for 2SOD geometry...")
    
    config = {
        "file": {
            "in_dir": "./input",
            "model_dir": "./model",
            "out_dir": "./output", 
            "voxel_size": 1.0,
            "SOD": 410,
            "SDD": 820,  # 2 × SOD for symmetric geometry
            "detector_geometry": "arc",
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
    
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)
    print(f"  ✅ Updated config.json with 2SOD parameters")
    
    # Visualize the prepared data
    print(f"\n📊 Visualizing prepared data...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0,0].imshow(slice42, cmap='gray')
    axes[0,0].set_title('Original Slice42')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(sino42, cmap='gray', aspect='auto')
    axes[0,1].set_title(f'Sinogram {sino42.shape}')
    axes[0,1].set_xlabel('Detector')
    axes[0,1].set_ylabel('Angle')
    
    if rec42 is not None:
        axes[1,0].imshow(rec42, cmap='gray')
        axes[1,0].set_title('Verification Reconstruction')
        axes[1,0].axis('off')
    
    axes[1,1].plot(fan_positions, 'b.-', markersize=1)
    axes[1,1].set_title('2SOD Fan Sensor Positions')
    axes[1,1].set_xlabel('Detector Index')
    axes[1,1].set_ylabel('Position (mm)')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n✅ 2SOD data preparation complete!")
    print(f"📁 Data saved to input/ folder")
    print(f"⚙️ Config updated for 2SOD symmetric geometry")
    print(f"\n🚀 You can now run:")
    print(f"   python main.py")
    print(f"   (or python main.py --img_id 0 for specific case)")
    
    return True

if __name__ == "__main__":
    prepare_2SOD_for_input()