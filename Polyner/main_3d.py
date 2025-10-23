# ----------------------------------------------#
# Pro    : cbct
# File   : main_3d.py
# Date   : 2025
# Author : 3D Cone-Beam CT Entry Point
# ----------------------------------------------#
import json
import Polyner_3d

if __name__ == '__main__':
    # Load 3D configuration
    with open('./config_3d.json', 'r') as f:
        config = json.load(f)

    # Image ID to process
    img_id = 0

    print("="*60)
    print("3D Cone-Beam CT Reconstruction with Polyner")
    print("="*60)
    print(f"Volume dimensions: {config['file']['h']}x{config['file']['w']}x{config['file']['d']}")
    print(f"SOD: {config['file']['SOD']} mm")
    print(f"SDD: {config['file']['SDD']} mm")
    print(f"Voxel size: {config['file']['voxel_size']} mm")
    print(f"Training epochs: {config['train']['epoch']}")
    print(f"Batch size: {config['train']['batch_size']}")
    print(f"Network neurons: {config['network']['n_neurons']}")
    print("="*60)

    # Train for specified image
    print(f"\nStarting 3D reconstruction for image #{img_id}...\n")
    network = Polyner_3d.train(img_id, config)

    print("\n" + "="*60)
    print("3D Reconstruction Complete!")
    print("="*60)
