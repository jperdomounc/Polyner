
## Project Overview

Polyner is a PyTorch implementation of "Unsupervised Polychromatic Neural Representation for CT Metal Artifact Reduction" (NeurIPS 2023). It's a deep learning project that uses neural representations to reduce metal artifacts in CT scans through polychromatic modeling.

## Key Commands

### Training
```bash
python main.py
```
Trains the Polyner model on metal-corrupted sinograms (ma_sinogram_0~9.nii). Models are saved in `./model/` and results in `./output/`.

### Evaluation
```bash
python eval.py
```
Computes PSNR and SSIM metrics comparing FBP reconstructions and Polyner results against ground truth.

### Data Simulation
Navigate to `./data_simulation/` and run:
```matlab
dl_data.m
```
Generates metal-corrupted measurements using MATLAB (requires ADN repository dependencies).

## Dependencies

- PyTorch 3.8.13
- tinycudann (TinyCUDA neural networks)
- SimpleITK
- tqdm, numpy
- commentjson
- scipy
- scikit-image
- skimage.morphology

## Architecture

### Core Components

**Training Pipeline (`Polyner.py`)**:
- `train(img_id, config)` - Main training function
- Uses TinyCUDA NetworkWithInputEncoding for neural representation
- Implements polychromatic forward model with energy spectrum
- Custom EAS (Attenuation Smoothing over Energies) loss function

**Network Architecture**:
- Hash-based multi-resolution encoding (Grid encoding)
- Fully-fused MLP with ReLU activation
- Outputs attenuation coefficients for multiple energy levels
- Configured via `config.json`

**Data Management (`dataset.py`)**:
- `TrainData` - Handles ray sampling and projection data
- `TestData` - Grid coordinates for reconstruction
- Fan-beam geometry ray generation

**Loss Function (`model.py`)**:
- `Attenuation_Smootion_Over_Energies_Loss` - Enforces smoothness across energy levels
- Combined with L1 data consistency loss

### Key Algorithms

**Forward Model**: Projects 3D attenuation through polychromatic X-ray spectrum
- Ray sampling from fan-beam geometry
- Energy-dependent attenuation modeling
- Exponential projection model with Beer-Lambert law

**Geometry Handling (`utils.py`)**:
- `fan_beam_ray()` - Generates ray paths for fan-beam CT
- `rotate_ray()` - Handles projection angle rotations
- `grid_coordinate()` - Creates reconstruction coordinate grids

## Configuration

All parameters are defined in `config.json`:
- **File paths**: input/output directories, voxel size, geometry
- **Training**: learning rate, epochs, batch size, lambda regularization
- **Encoding**: hash grid parameters, resolution levels
- **Network**: MLP architecture, activation functions

## Data Structure

**Input Files** (`./input/`):
- `gt_x.nii` - Ground truth images
- `ma_sinogram_x.nii` - Metal-corrupted sinograms
- `mask_x.nii` - Metal masks
- `ma_x.nii` - FBP reconstructions
- `fanSensorPos.nii` - Fan-beam detector positions
- `GE14Spectrum120KVP.mat` - X-ray energy spectrum

**Output Files** (`./output/`):
- `polyner_x.nii` - Neural reconstruction results

## Development Notes

- The project processes 10 sample images (indexed 0-9)
- Uses NIFTI format (.nii) for medical images - viewable with ITK-SNAP
- MATLAB simulation code in `data_simulation/` requires separate setup
- Energy spectrum modeling uses 20-120 keV range from GE scanner data
- Training uses CUDA if available, falls back to CPU