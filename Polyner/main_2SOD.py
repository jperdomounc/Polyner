#!/usr/bin/env python3
"""
Main training script for 2SOD symmetric geometry dataset
"""

import Polyner
import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Train Polyner on 2SOD symmetric geometry dataset')
    parser.add_argument('--config', type=str, default='config_2SOD.json', 
                       help='Configuration file path')
    parser.add_argument('--img_id', type=int, default=0, 
                       help='Image ID to train on')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Verify 2SOD geometry
    SOD = config['file']['SOD']
    SDD = config['file']['SDD']
    
    print("🎯 2SOD Symmetric Geometry Configuration:")
    print(f"   SOD: {SOD} mm")
    print(f"   SDD: {SDD} mm")
    print(f"   Ratio: {SDD/SOD:.1f} (should be ~2.0)")
    print(f"   Geometry: {config['file'].get('geometry_type', 'symmetric_fan_beam')}")
    print(f"   Detector: {config['file'].get('detector_geometry', 'arc')}")
    
    # Verify input files exist
    input_dir = Path(config['file']['in_dir'])
    required_files = [
        f"gt_{args.img_id}.nii",
        f"ma_sinogram_{args.img_id}.nii", 
        f"mask_{args.img_id}.nii",
        "fanSensorPos.nii"
    ]
    
    print(f"\n📁 Checking input files in {input_dir}:")
    all_files_exist = True
    for file_name in required_files:
        file_path = input_dir / file_name
        if file_path.exists():
            print(f"   ✅ {file_name}")
        else:
            print(f"   ❌ {file_name} - MISSING")
            all_files_exist = False
    
    if not all_files_exist:
        print("\n❌ Missing required files. Please run prepare_2SOD_data.py first.")
        return
    
    # Create output directories
    Path(config['file']['model_dir']).mkdir(exist_ok=True)
    Path(config['file']['out_dir']).mkdir(exist_ok=True)
    
    # Start training
    print(f"\n🚀 Starting Polyner training for image {args.img_id}...")
    print(f"   Epochs: {config['train']['epoch']}")
    print(f"   Learning rate: {config['train']['lr']}")
    print(f"   Batch size: {config['train']['batch_size']}")
    
    try:
        Polyner.train(args.img_id, config)
        print(f"\n✅ Training completed successfully!")
        print(f"   Model saved in: {config['file']['model_dir']}")
        print(f"   Results saved in: {config['file']['out_dir']}")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()