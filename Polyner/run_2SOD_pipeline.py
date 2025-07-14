#!/usr/bin/env python3
"""
Complete 2SOD Pipeline Script
Run this once after cloning the repo in Google Colab
"""

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
import json
import shutil
import subprocess
import sys

def install_dependencies():
    """Install required packages"""
    print("🔧 Installing dependencies...")
    packages = [
        "torch torchvision torchaudio",
        "ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch",
        "SimpleITK tqdm numpy scipy scikit-image commentjson matplotlib"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + package.split())
        except:
            print(f"⚠️ Warning: Could not install {package}")
    print("✅ Dependencies installed")

def load_binary_data(file_path, shape, dtype=np.float32):
    """Load binary data and reshape."""
    data = np.fromfile(str(file_path), dtype=dtype)
    return data.reshape(shape)

def save_as_nifti(data, output_path):
    """Save numpy array as NIFTI file."""
    if data.ndim == 2:
        data = data.astype(np.float32)
    img = sitk.GetImageFromArray(data)
    sitk.WriteImage(img, str(output_path))
    return img

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
        Path("unctestdata/2sod"),
        Path("2SOD"),
        Path("2sod")
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    return None

def load_2SOD_data():
    """Load 2SOD dataset files."""
    print("🎯 Loading 2SOD dataset...")
    
    # Create directories
    Path('input_2SOD').mkdir(exist_ok=True)
    Path('output_2SOD').mkdir(exist_ok=True)
    Path('model_2SOD').mkdir(exist_ok=True)
    
    # Find data directory
    data_base = find_2SOD_directory()
    if data_base is None:
        print("❌ Could not find 2SOD data directory")
        return None, None, None
    
    print(f"✅ Found data directory: {data_base}")
    
    slice42 = None
    sino42 = None
    rec42 = None
    
    # Load slice42_sod
    slice_file = data_base / "slice42_sod"
    if slice_file.exists():
        try:
            slice42 = load_binary_data(slice_file, (216, 216))
            save_as_nifti(slice42, 'input_2SOD/gt_0.nii')
            print(f"✅ Loaded slice42_sod: {slice42.shape}")
        except:
            try:
                slice42 = load_binary_data(slice_file, (216, 216), dtype=np.float64)
                save_as_nifti(slice42.astype(np.float32), 'input_2SOD/gt_0.nii')
                print(f"✅ Loaded slice42_sod (float64): {slice42.shape}")
            except Exception as e:
                print(f"❌ Failed to load slice42_sod: {e}")
    
    # Load sino42_sod
    sino_file = data_base / "sino42_sod"
    if sino_file.exists():
        file_size = sino_file.stat().st_size
        
        # Try different combinations
        attempts = [
            ((360, 400), np.float32),
            ((400, 360), np.float32),
            ((360, 400), np.float64),
            ((400, 360), np.float64),
            ((744, 360), np.float32),
            ((360, 744), np.float32)
        ]
        
        for shape, dtype in attempts:
            try:
                sino42 = load_binary_data(sino_file, shape, dtype)
                # Ensure (angles, detectors) format
                if shape[0] > shape[1]:
                    sino42 = sino42.T
                save_as_nifti(sino42.astype(np.float32), 'input_2SOD/ma_sinogram_0.nii')
                print(f"✅ Loaded sino42_sod: {sino42.shape}")
                break
            except:
                continue
        
        if sino42 is None:
            print(f"❌ Could not load sino42_sod (size: {file_size} bytes)")
    
    # Load rec42_sod
    rec_file = data_base / "rec42_sod"
    if rec_file.exists():
        try:
            rec42 = load_binary_data(rec_file, (216, 216))
            save_as_nifti(rec42, 'input_2SOD/ma_0.nii')
            print(f"✅ Loaded rec42_sod: {rec42.shape}")
        except:
            try:
                rec42 = load_binary_data(rec_file, (216, 216), dtype=np.float64)
                save_as_nifti(rec42.astype(np.float32), 'input_2SOD/ma_0.nii')
                print(f"✅ Loaded rec42_sod (float64): {rec42.shape}")
            except Exception as e:
                print(f"❌ Failed to load rec42_sod: {e}")
    
    return slice42, sino42, rec42

def create_additional_files(slice42):
    """Create fan sensor positions and metal mask."""
    # Generate fan sensor positions
    fan_positions = generate_2SOD_fan_positions(400, 410)
    fanSensorPos = fan_positions.reshape(-1, 1)
    fanPos_sitk = sitk.GetImageFromArray(fanSensorPos)
    sitk.WriteImage(fanPos_sitk, 'input_2SOD/fanSensorPos.nii')
    print(f"✅ Fan positions generated")
    
    # Create metal mask
    if slice42 is not None:
        threshold = np.percentile(slice42, 95)
        metal_mask = (slice42 > threshold).astype(np.float32)
        save_as_nifti(metal_mask, 'input_2SOD/mask_0.nii')
        print(f"✅ Metal mask created")
        return metal_mask
    return None

def copy_spectrum():
    """Copy spectrum file."""
    spectrum_sources = [
        'input/GE14Spectrum120KVP.mat',
        'input/spectrum_UNC.mat', 
        'UNCtestdata/spectrum_UNC.mat',
        'GE14Spectrum120KVP.mat'
    ]
    
    for spectrum_file in spectrum_sources:
        if Path(spectrum_file).exists():
            shutil.copy2(spectrum_file, f'input_2SOD/{Path(spectrum_file).name}')
            print(f"✅ Copied spectrum: {Path(spectrum_file).name}")
            return True
    print("⚠️ No spectrum file found")
    return False

def create_config():
    """Create 2SOD configuration."""
    config_2SOD = {
        "file": {
            "in_dir": "./input_2SOD",
            "model_dir": "./model_2SOD", 
            "out_dir": "./output_2SOD",
            "voxel_size": 1.0,
            "SOD": 410,
            "SDD": 820,
            "detector_geometry": "arc",
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
    
    with open('config_2SOD.json', 'w') as f:
        json.dump(config_2SOD, f, indent=4)
    print("✅ Configuration created")
    return config_2SOD

def visualize_data(slice42, sino42, rec42, metal_mask, fan_positions):
    """Visualize the loaded data."""
    if slice42 is None:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    axes[0,0].imshow(slice42, cmap='gray')
    axes[0,0].set_title('Original Slice42')
    axes[0,0].axis('off')
    
    if sino42 is not None:
        axes[0,1].imshow(sino42, cmap='gray', aspect='auto')
        axes[0,1].set_title(f'Sinogram {sino42.shape}')
        axes[0,1].set_xlabel('Detector')
        axes[0,1].set_ylabel('Angle')
    
    if rec42 is not None:
        axes[0,2].imshow(rec42, cmap='gray')
        axes[0,2].set_title('Verification')
        axes[0,2].axis('off')
    
    if metal_mask is not None:
        axes[1,0].imshow(metal_mask, cmap='hot')
        axes[1,0].set_title('Metal Mask')
        axes[1,0].axis('off')
    
    axes[1,1].plot(fan_positions, 'b.-', markersize=1)
    axes[1,1].set_title('Fan Sensor Positions')
    axes[1,1].set_xlabel('Detector Index')
    axes[1,1].set_ylabel('Position (mm)')
    axes[1,1].grid(True, alpha=0.3)
    
    axes[1,2].plot(slice42[108, :], 'g-', label='Original', linewidth=2)
    if rec42 is not None:
        axes[1,2].plot(rec42[108, :], 'r--', label='Verification', alpha=0.7)
    axes[1,2].set_title('Center Row Profile')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def run_training():
    """Run Polyner training."""
    required_files = [
        'input_2SOD/gt_0.nii', 
        'input_2SOD/ma_sinogram_0.nii', 
        'input_2SOD/mask_0.nii', 
        'input_2SOD/fanSensorPos.nii'
    ]
    
    print(f"\n📋 Checking required files:")
    all_files_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            all_files_exist = False
    
    if not all_files_exist:
        print("❌ Missing files - cannot start training")
        return False
    
    print(f"\n🚀 Starting Polyner training on 2SOD symmetric geometry...")
    try:
        result = subprocess.run([
            sys.executable, "main.py", 
            "--config", "config_2SOD.json", 
            "--img_id", "0"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Training completed successfully!")
            return True
        else:
            print(f"❌ Training failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False

def evaluate_results(slice42, rec42):
    """Evaluate and visualize results."""
    if not Path('output_2SOD/polyner_0.nii').exists():
        print("❌ No Polyner results found")
        return
    
    polyner_result = sitk.GetArrayFromImage(sitk.ReadImage('output_2SOD/polyner_0.nii'))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    if slice42 is not None:
        axes[0,0].imshow(slice42, cmap='gray')
        axes[0,0].set_title('Original')
        axes[0,0].axis('off')
    
    if rec42 is not None:
        axes[0,1].imshow(rec42, cmap='gray')
        axes[0,1].set_title('Verification')
        axes[0,1].axis('off')
    
    axes[0,2].imshow(polyner_result, cmap='gray')
    axes[0,2].set_title('Polyner Result')
    axes[0,2].axis('off')
    
    if rec42 is not None:
        diff = np.abs(polyner_result - rec42)
        axes[1,0].imshow(diff, cmap='hot')
        axes[1,0].set_title('|Polyner - Verification|')
        axes[1,0].axis('off')
    
    # Profiles
    center_row = 108
    axes[1,1].plot(polyner_result[center_row, :], 'b-', label='Polyner', linewidth=2)
    if slice42 is not None:
        axes[1,1].plot(slice42[center_row, :], 'g--', label='Original', alpha=0.7)
    if rec42 is not None:
        axes[1,1].plot(rec42[center_row, :], 'r:', label='Verification', alpha=0.7)
    axes[1,1].set_title('Center Row Profile')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # Metrics
    if rec42 is not None:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        psnr = peak_signal_noise_ratio(rec42, polyner_result, data_range=rec42.max()-rec42.min())
        ssim = structural_similarity(rec42, polyner_result, data_range=rec42.max()-rec42.min())
        
        axes[1,2].text(0.1, 0.8, f'PSNR: {psnr:.2f} dB', transform=axes[1,2].transAxes, fontsize=12)
        axes[1,2].text(0.1, 0.6, f'SSIM: {ssim:.4f}', transform=axes[1,2].transAxes, fontsize=12)
        axes[1,2].text(0.1, 0.4, f'SOD: 410 mm', transform=axes[1,2].transAxes, fontsize=10)
        axes[1,2].text(0.1, 0.2, f'SDD: 820 mm (2×SOD)', transform=axes[1,2].transAxes, fontsize=10)
        axes[1,2].set_title('2SOD Results')
        axes[1,2].axis('off')
        
        print(f"\n📊 Results vs Verification:")
        print(f"   PSNR: {psnr:.2f} dB")
        print(f"   SSIM: {ssim:.4f}")
    
    plt.tight_layout()
    plt.show()

def package_results():
    """Package results for download."""
    try:
        subprocess.run(['zip', '-r', '2SOD_results.zip', 'output_2SOD/', 'model_2SOD/', 'config_2SOD.json'], 
                      check=True)
        print("✅ Results packaged in 2SOD_results.zip")
    except:
        print("⚠️ Could not create zip file")

def main():
    """Main pipeline execution."""
    print("🎯 2SOD Symmetric Geometry Pipeline Starting...")
    
    # Install dependencies
    install_dependencies()
    
    # Load data
    slice42, sino42, rec42 = load_2SOD_data()
    
    # Create additional files
    metal_mask = create_additional_files(slice42)
    fan_positions = generate_2SOD_fan_positions(400, 410)
    
    # Copy spectrum and create config
    copy_spectrum()
    config = create_config()
    
    # Visualize data
    visualize_data(slice42, sino42, rec42, metal_mask, fan_positions)
    
    # Run training
    if run_training():
        # Evaluate results
        evaluate_results(slice42, rec42)
        # Package results
        package_results()
        print("\n🎉 2SOD Pipeline completed successfully!")
    else:
        print("\n❌ Pipeline failed during training")

if __name__ == "__main__":
    main()