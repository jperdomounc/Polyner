# 2SOD Symmetric Geometry Testing Pipeline

This pipeline tests the Polyner metal artifact reduction on the 2SOD symmetric geometry dataset where **SDD = 2 × SOD**.

## Dataset Overview

The 2SOD symmetric geometry dataset contains:
- `slice42` - Original slice (216×216)
- `sino42` - Sinogram (360×400 or 400×360) 
- `rec42` - Verification reconstruction (216×216)
- MATLAB code for fan-beam projection
- Geometry parameters: SOD = 410mm, SDD = 820mm (2×SOD)

## Files Created

### 🔧 Core Pipeline Files
- `prepare_2SOD_data.py` - Data loading and conversion script
- `main_2SOD.py` - Training script for 2SOD geometry
- `eval_2SOD.py` - Evaluation and comparison script
- `config_2SOD.json` - Configuration file (auto-generated)

### 📓 Google Colab Integration
- `colab_2SOD_pipeline.ipynb` - Complete Colab notebook for cloud execution

## Usage Instructions

### Local Execution

1. **Data Setup**
   ```bash
   # Create data directory and place 2SOD files
   mkdir 2SOD_data
   # Place files: slice42*.bin, sino42*.bin, rec42*.bin
   ```

2. **Data Preparation**
   ```bash
   python prepare_2SOD_data.py
   ```
   This creates:
   - `input_2SOD/` directory with converted NIFTI files
   - `config_2SOD.json` with symmetric geometry parameters

3. **Training**
   ```bash
   python main_2SOD.py --config config_2SOD.json --img_id 0
   ```

4. **Evaluation**
   ```bash
   python eval_2SOD.py --config config_2SOD.json --img_id 0 --save_plots
   ```

### Google Colab Execution

1. Upload `colab_2SOD_pipeline.ipynb` to Google Colab
2. Upload 2SOD data files when prompted
3. Run all cells sequentially
4. Download results as `2SOD_results.zip`

## Configuration Details

### 2SOD Symmetric Geometry Parameters
```json
{
  "file": {
    "SOD": 410,           // Source-to-object distance (mm)
    "SDD": 820,           // Source-to-detector distance (2×SOD)
    "detector_geometry": "arc",  // Arc detector for symmetric geometry
    "geometry_type": "symmetric_fan_beam",
    "h": 216, "w": 216    // Image dimensions
  }
}
```

### Key Features
- **Symmetric Geometry**: SDD = 2 × SOD ensures symmetric fan-beam reconstruction
- **Arc Detector**: Uses arc detector geometry (not linear)
- **Verification**: Compares against provided rec42 reconstruction
- **Metrics**: PSNR, SSIM, MSE, MAE calculations

## Expected Outputs

### Training Results
- `model_2SOD/model_0.pkl` - Trained Polyner model
- `output_2SOD/polyner_0.nii` - Polyner reconstruction

### Evaluation Results  
- `output_2SOD/metrics_0.json` - Quantitative metrics
- `output_2SOD/comparison_0.png` - Visual comparison plots
- Console output with PSNR/SSIM values

### Comparison Metrics
- **Polyner vs Original**: Quality of neural reconstruction
- **Polyner vs Verification**: Consistency with provided reconstruction
- **Visual Analysis**: Side-by-side images, profiles, histograms

## File Format Requirements

The pipeline expects binary files in these formats:
- `slice42_216_216.bin` - Float32, 216×216 pixels
- `sino42_400_360.bin` - Float32, 360×400 (angles×detectors) 
- `rec42_216_216.bin` - Float32, 216×216 pixels

Alternative naming patterns are supported:
- `slice42.bin`, `sino42.bin`, `rec42.bin`

## Troubleshooting

### Common Issues
1. **File not found**: Ensure 2SOD data files are in `2SOD_data/` directory
2. **Shape mismatch**: Pipeline auto-detects sinogram orientation
3. **Missing spectrum**: Uses default GE spectrum if UNC spectrum unavailable
4. **GPU memory**: Reduce batch size in config if CUDA out of memory

### Dependencies
```bash
pip install torch torchvision torchaudio
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install SimpleITK tqdm numpy scipy scikit-image matplotlib commentjson
```

## Branch Information

This pipeline is developed in the `2SOD-geometry` branch specifically for testing the symmetric geometry dataset. The implementation preserves all original Polyner functionality while adapting for the 2SOD geometry constraints.

## Results Interpretation

### Good Results Indicators
- PSNR > 30 dB vs verification reconstruction
- SSIM > 0.95 vs verification reconstruction  
- Visual consistency in comparison plots
- Smooth convergence during training

### Quality Metrics
- **PSNR**: Peak Signal-to-Noise Ratio (higher = better)
- **SSIM**: Structural Similarity Index (closer to 1 = better)
- **MSE/MAE**: Mean Squared/Absolute Error (lower = better)
- **Correlation**: Pixel-wise correlation with verification

This pipeline provides a complete testing framework for validating Polyner performance on symmetric geometry datasets with comprehensive evaluation and visualization capabilities.