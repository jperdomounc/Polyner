# Metal Artifact Reduction Package for UNC Multisource Array CBCT System

This repository provides a metal artifact reduction (MAR) package specifically adapted for the University of North Carolina's multisource array cone-beam CT (CBCT) system. The package is based on the Polyner method from the NeurIPS 2023 paper "*Unsupervised Polychromatic Neural Representation for CT Metal Artifact Reduction*" by Wu et al. [[OpenReview](https://openreview.net/forum?id=xx3QgKyghS)], [[arXiv](https://arxiv.org/abs/2306.15203)]

The implementation has been specifically modified to work with UNC's linear detector geometry and multisource array configuration, providing effective metal artifact reduction for clinical and research applications.

![image](gif/fig_method.jpg)
*Fig. 1: Overview of the Polyner model architecture adapted for UNC CBCT system.*

## 1. Visualization

![image](gif/fig1.gif)
*Fig. 2: Qualitative results of FBP and our polyner on 2D fan-beam samples of DeepLesion simulation dataset.*

![image](gif/fig2.gif)
*Fig. 3: Qualitative results of FDK and our polyner on a real-world 3D cone-beam mouse thigh sample.*
## 2. File Tree
```
PolynerCode
│  readme.md					# this readme file
│  notes.txt					# development notes
│
├─Polyner					# main package directory
│  │  config.json				# 2D configuration (original)
│  │  config_3d.json				# 3D cone-beam CT configuration
│  │  main_3d.py				# 3D CBCT training script
│  │  Polyner_3d.py				# 3D training function
│  │  dataset_3d.py				# 3D dataloader
│  │  utils_3d.py				# 3D reconstruction utilities
│  │  model.py					# Attenuation Smoothness over Energies (ASE) loss
│  │  generate_test_data_3d.py			# 3D test data generator
│  │  notes.txt					# development notes
│  │
│  └─input					# 3D CBCT input data
│          ma_projection_0.nii			# metal-corrupted projections
│          mask_0.nii				# metal masks
│          detectorUPos.nii			# detector U (horizontal) positions
│          detectorVPos.nii			# detector V (vertical) positions
│          DECBCTSpectrum110KVP.mat		# UNC X-ray energy spectrum
```

## 3. Dependencies and Requirements

### Python Version
- **Python 3.9+** (recommended for PyTorch 2.x compatibility)
- Tested with Python 3.9.6

### Python Dependencies
The following Python packages are required to run the 3D CBCT MAR package:

**Core Dependencies:**
- **PyTorch 2.7+** (with CUDA support strongly recommended for GPU acceleration)
- **tinycudann** (tiny-cuda-nn) - NVIDIA's tiny-cuda-nn for neural network acceleration
- **numpy 2.0+** - Numerical computing
- **SimpleITK** - Medical image I/O and processing
- **scipy 1.13+** - Scientific computing library (for .mat file I/O)

**Additional Dependencies:**
- **tqdm 4.67+** - Progress bars for training
- **scikit-image 0.24+** - Image processing metrics (SSIM, PSNR) and morphological operations

**Standard Library (included with Python):**
- json - Configuration file parsing
- pathlib - Path handling
- time - Performance timing

### System Requirements
- **CUDA-compatible NVIDIA GPU** (required for tinycudann)
- **CUDA Toolkit 11.x or 12.x**
- Minimum 8GB GPU VRAM (16GB+ recommended for large volumes)
- Minimum 16GB system RAM (32GB+ recommended)
- Storage space for datasets and models

### Installation

**Step 1: Install PyTorch with CUDA support**
```bash
# For CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Step 2: Install tiny-cuda-nn**
```bash
pip3 install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

**Step 3: Install remaining dependencies**
```bash
pip3 install numpy scipy SimpleITK tqdm scikit-image
```

**Verify Installation:**
```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import tinycudann; print('tinycudann installed successfully')"
```

## 4. UNC Multisource Array CBCT Configuration

This package has been specifically adapted for UNC's multisource array CBCT system with the following specifications:

### System Geometry
- **Detector Type**: Linear detector array (not arc geometry)
- **Source-to-Object Distance (SOD)**: 410mm
- **Source-to-Detector Distance (SDD)**: 620mm  
- **Detector Dimensions**: 148.8mm × 148.8mm
- **Detector Pixel Size**: 0.2mm
- **Detector Offset**: 70.5mm

### UNC-Specific Data Processing
- **Input Data**: RANDO phantom with metal implants
- **Data Format**: NIfTI format (.nii) for projections, volumes, and masks
- **Energy Spectrum**: UNC X-ray spectrum data (`DECBCTSpectrum110KVP.mat`)
- **Geometry Configuration**: 3D cone-beam geometry with flat panel detector

### Training for 3D CBCT Data
```bash
# Navigate to the Polyner directory
cd Polyner

# Train 3D CBCT reconstruction
python3 main_3d.py
```

The 3D-specific configuration file (`config_3d.json`) contains parameters optimized for the cone-beam geometry and UNC clinical protocols.

## 5. Training and 3D Reconstruction

To train the 3D cone-beam CT reconstruction model:

```bash
cd Polyner
python3 main_3d.py
```

**Configuration:** Edit `config_3d.json` to customize:
- Volume dimensions (h, w, d)
- Training epochs and batch size
- Network architecture parameters
- Source-to-object distance (SOD) and source-to-detector distance (SDD)
- Input/output directories

**Training Output:**
- Models are saved to the directory specified in `config_3d.json` (default: `./model`)
- Reconstructed 3D volumes (.nii files) are saved during training at intervals specified by `save_epoch`
- Output filename format: `polyner_3d_{img_id}_{loss}_{iterations_per_sec}it_s.nii`

**Training Progress:**
The script displays:
- Current epoch and loss
- Learning rate
- Iterations per second
- Progress bar via tqdm

## 6. Viewing Results

**Viewing 3D Volumes:**
NIfTI files (`.nii`) can be viewed using ITK-SNAP, 3D Slicer, or other medical imaging software:
- **ITK-SNAP**: http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.SNAP4
- **3D Slicer**: https://www.slicer.org/

**Evaluation Metrics:**
The `utils_3d.py` module provides functions for computing:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)

These metrics can be used to quantitatively evaluate reconstruction quality against ground truth when available.


## 7. Key Implementation Details

### 3D Cone-Beam Geometry
The implementation handles 3D cone-beam CT with:
- **Flat panel detector** (not arc geometry)
- **3D ray generation** through the reconstruction volume
- **Gantry rotation** around the z-axis (configurable)
- **Separate U and V detector coordinates** for horizontal and vertical directions

### Neural Network Architecture
- **Input**: 3D spatial coordinates (x, y, z) in normalized space [-1, 1]
- **Encoding**: Configurable hash encoding for efficient 3D representation
- **Output**: Attenuation coefficients at multiple energy levels
- **Framework**: tiny-cuda-nn for high-performance training

### Forward Model
- **Polyenergetic X-ray spectrum**: 110 kVp spectrum from UNC system
- **Beer's Law**: Line integral along rays through the volume
- **Energy weighting**: Spectrum-weighted integration for realistic projections

### Loss Functions
1. **Data Consistency Loss (L1)**: Matches predicted projections to measured projections
2. **Attenuation Smoothness over Energies (ASE) Loss**: Enforces physical constraint that metal regions have similar attenuation across energies

## 8. Technical Notes

### Memory Considerations
- Large 3D volumes require significant GPU memory
- Batch size and num_sample_ray can be reduced if GPU memory is limited
- Consider processing smaller sub-volumes for very large datasets

### Training Tips
- Training typically requires 1000-5000 epochs depending on volume complexity
- Learning rate scheduling helps convergence
- Monitor both data consistency and ASE loss components
- Save checkpoints regularly (default: every 100 epochs)

### Coordinate System
- Origin at volume center
- Normalized coordinates: [-1, 1] in all dimensions
- Source positioned along negative y-axis
- Detector positioned along positive y-axis
- Gantry rotates around z-axis

## 9. Troubleshooting

### Common Issues

**"tinycudann not found"**
- Ensure CUDA Toolkit is installed
- Check PyTorch CUDA compatibility
- Rebuild tiny-cuda-nn from source if needed

**"CUDA out of memory"**
- Reduce batch_size in config_3d.json
- Reduce num_sample_ray (number of rays per batch)
- Reduce num_samples (samples per ray)
- Use smaller volume dimensions

**"No module named 'SimpleITK'"**
```bash
pip3 install SimpleITK
```

**Slow training performance**
- Verify GPU is being used: Check CUDA availability
- Ensure tiny-cuda-nn is properly installed with CUDA support
- Monitor GPU utilization with `nvidia-smi`

## 10. License

This code is available for non-commercial research and education purposes only. It is not allowed to be reproduced, exchanged, sold, or used for profit.

## 11. Citation

The original code and paper was completed by the following people below:
```
@inproceedings{
wu2023unsupervised,
title={Unsupervised Polychromatic Neural Representation for {CT} Metal Artifact Reduction},
author={Qing Wu and Lixuan Chen and Ce Wang and Hongjiang Wei and S Kevin Zhou and Jingyi Yu and Yuyao Zhang},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=xx3QgKyghS}
}
```
