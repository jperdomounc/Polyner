#!/usr/bin/env python3
"""
Prepare 2SOD data for Polyner input
Converts UNCtestdata/2SOD binary files to input/ NIFTI format
"""

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path

def load_binary_data(file_path, shape, dtype=np.float32):
    """Load binary data and reshape."""
    print(f"  Loading: {file_path} -> {shape}")
    data = np.fromfile(str(file_path), dtype=dtype)
    return data.reshape(shape)

def save_as_nifti(data, output_path):
    """Save numpy array as NIFTI file."""
    data = data.astype(np.float32)
    img = sitk.GetImageFromArray(data)
    sitk.WriteImage(img, str(output_path))
    print(f"  ✅ Saved: {output_path}")

def generate_2SOD_fan_positions(num_detectors=520, SOD=410):
    """Generate fan sensor positions for symmetric 2SOD geometry."""
    # Calculate detector angle range
    gamma_max = np.arctan(108 / SOD)  # Half fan angle
    detector_angles = np.linspace(-gamma_max, gamma_max, num_detectors)
    
    # For symmetric geometry, SDD = 2 * SOD
    SDD = 2 * SOD  # 820 mm
    detector_positions = SDD * np.tan(detector_angles)
    
    return detector_positions.astype(np.float32)

def prepare_2SOD_data():
    """Convert 2SOD data to input/ format."""
    print("🎯 Preparing 2SOD data for Polyner input...")
    
    # Paths
    data_dir = Path("UNCtestdata/2SOD")
    input_dir = Path("input")
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    print(f"✅ Found data directory: {data_dir}")
    
    # Load slice42_2SOD.bin (original slice)
    slice_file = data_dir / "slice42_2SOD.bin"
    print(f"\n📂 Loading slice42_2SOD.bin...")
    try:
        slice42 = load_binary_data(slice_file, (216, 216), np.float32)
        print(f"  Range: [{np.min(slice42):.3f}, {np.max(slice42):.3f}]")
        save_as_nifti(slice42, input_dir / "gt_0.nii")
    except Exception as e:
        print(f"  ❌ Failed to load slice42: {e}")
        return False
    
    # Load sino42_2SOD.bin (sinogram)
    sino_file = data_dir / "sino42_2SOD.bin"
    print(f"\n📂 Loading sino42_2SOD.bin...")
    file_size = sino_file.stat().st_size
    print(f"  File size: {file_size} bytes")
    
    # Calculate possible shapes
    total_float32 = file_size // 4
    total_float64 = file_size // 8
    
    print(f"  Total elements: {total_float32} (float32) or {total_float64} (float64)")
    
    # Try different sinogram shapes
    sino42 = None
    possible_shapes = [
        ((360, 520), np.float32),  # 360 angles, 520 detectors
        ((520, 360), np.float32),  # 520 angles, 360 detectors
        ((360, 520), np.float64),
        ((520, 360), np.float64),
        ((400, 468), np.float32),  # Alternative dimensions
        ((468, 400), np.float32),
    ]
    
    for (shape, dtype) in possible_shapes:
        if (shape[0] * shape[1]) == (total_float32 if dtype == np.float32 else total_float64):
            try:
                print(f"  Trying {shape} {dtype.__name__}...")
                sino42 = load_binary_data(sino_file, shape, dtype)
                
                # Ensure (angles, detectors) format - typically angles < detectors
                if shape[0] > shape[1]:
                    print(f"    Transposing from {sino42.shape} to {sino42.T.shape}")
                    sino42 = sino42.T
                
                print(f"  ✅ Success: {sino42.shape}, range: [{np.min(sino42):.3f}, {np.max(sino42):.3f}]")
                save_as_nifti(sino42, input_dir / "ma_sinogram_0.nii")
                break
            except Exception as e:
                print(f"    Failed: {e}")
                continue
    
    if sino42 is None:
        print("  ❌ Could not load sinogram")
        return False
    
    # Load rec42_2SOD.bin (verification reconstruction)
    rec_file = data_dir / "rec42_2SOD.bin"
    print(f"\n📂 Loading rec42_2SOD.bin...")
    try:
        rec42 = load_binary_data(rec_file, (216, 216), np.float32)
        print(f"  Range: [{np.min(rec42):.3f}, {np.max(rec42):.3f}]")
        save_as_nifti(rec42, input_dir / "ma_0.nii")
    except Exception as e:
        print(f"  ❌ Failed to load rec42: {e}")
        return False
    
    # Generate metal mask from slice42 (threshold at 95th percentile)
    print(f"\n🎭 Creating metal mask...")
    threshold = np.percentile(slice42, 95)
    metal_mask = (slice42 > threshold).astype(np.float32)
    save_as_nifti(metal_mask, input_dir / "mask_0.nii")
    print(f"  Metal pixels: {np.sum(metal_mask):.0f}, threshold: {threshold:.3f}")
    
    # Generate fan sensor positions for 2SOD geometry
    print(f"\n🔧 Creating 2SOD fan sensor positions...")
    num_detectors = sino42.shape[1]  # Use actual sinogram detector count
    print(f"  Sinogram shape: {sino42.shape} -> using {num_detectors} detectors")
    
    fan_positions = generate_2SOD_fan_positions(num_detectors, 410)
    fanSensorPos = fan_positions.reshape(-1, 1)
    fanPos_sitk = sitk.GetImageFromArray(fanSensorPos)
    sitk.WriteImage(fanPos_sitk, str(input_dir / "fanSensorPos.nii"))
    print(f"  ✅ Fan positions: {len(fan_positions)} detectors")
    print(f"  Range: [{np.min(fan_positions):.1f}, {np.max(fan_positions):.1f}] mm")
    
    # Update config.json for 2SOD geometry
    print(f"\n⚙️ Updating config.json for 2SOD geometry...")
    import json
    
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
    
    # Visualize prepared data
    print(f"\n📊 Visualizing prepared data...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0,0].imshow(slice42, cmap='gray')
    axes[0,0].set_title('Original Slice42')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(sino42, cmap='gray', aspect='auto')
    axes[0,1].set_title(f'Sinogram {sino42.shape}')
    axes[0,1].set_xlabel('Detector')
    axes[0,1].set_ylabel('Angle')
    
    axes[1,0].imshow(rec42, cmap='gray')
    axes[1,0].set_title('Verification Reconstruction')
    axes[1,0].axis('off')
    
    axes[1,1].plot(fan_positions, 'b.-', markersize=1)
    axes[1,1].set_title('2SOD Fan Sensor Positions')
    axes[1,1].set_xlabel('Detector Index')
    axes[1,1].set_ylabel('Position (mm)')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('2SOD_data_preparation.png', dpi=150, bbox_inches='tight')
    print("  ✅ Visualization saved as 2SOD_data_preparation.png")
    
    print(f"\n✅ 2SOD data preparation complete!")
    print(f"📁 Ready for testing with:")
    print(f"   python main.py --img_id 0")
    
    return True

if __name__ == "__main__":
    prepare_2SOD_data()