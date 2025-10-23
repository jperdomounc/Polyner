# Polyner 3D Cone-Beam CT Adaptation

This directory contains the 3D cone-beam CT adaptation of the Polyner model for metal artifact reduction (MAR).

## Overview

The original Polyner model was designed for 2D fan-beam CT reconstruction. This 3D version adapts it for cone-beam CT (CBCT) geometry, as described in the authors' response to reviewers in the OpenReview discussion.

### Key Reference

From the paper's reviewer response:
> "In the case of 3D cone-beam settings, only a simple modification following 3D acquisition geometry for X-ray simulation needs to be conducted, while all other steps remain unchanged."

The authors demonstrated this on a mouse CT scan (200×200×150 voxels) which required ~10 GB of memory and ~32 minutes on an NVIDIA RTX TITAN GPU.

## Files

### 3D Implementation Files (New)
- **`utils_3d.py`** - 3D geometry functions for cone-beam CT
  - `cone_beam_ray()` - Generates 3D rays for cone-beam geometry
  - `rotate_ray_3d()` - 3D rotation for gantry angles
  - `grid_coordinate_3d()` - 3D reconstruction volume grid
  - Also includes original 2D functions for backward compatibility

- **`dataset_3d.py`** - 3D data loaders
  - `TrainData3D` - Training dataset for 3D cone-beam projections
  - `TestData3D` - 3D volume reconstruction dataset
  - Also includes original 2D loaders

- **`Polyner_3d.py`** - Main 3D training loop
  - Adapted from original `Polyner.py`
  - Uses 3D network (n_input_dims=3)
  - Handles 3D projections and volume reconstruction

- **`config_3d.json`** - Configuration for 3D reconstruction
  - Volume dimensions (h, w, d)
  - SOD (Source-to-Origin Distance)
  - SDD (Source-to-Detector Distance)
  - Network parameters optimized for 3D

- **`main_3d.py`** - Entry point for 3D reconstruction

### Original 2D Files (Unchanged)
- `utils.py` - Original 2D geometry
- `dataset.py` - Original 2D data loaders
- `Polyner.py` - Original 2D training
- `config.json` - Original 2D config
- `main.py` - Original 2D entry point
- `model.py` - Loss functions (shared between 2D and 3D)

## Key Changes from 2D to 3D

### 1. **Geometry (utils_3d.py)**

#### 2D Fan-Beam → 3D Cone-Beam
```python
# 2D: fan_beam_ray(proj_pos, SOD)
# Returns: (num_det, 2*SOD, 2) - detector × samples × xy

# 3D: cone_beam_ray(detector_u_pos, detector_v_pos, SOD, SDD)
# Returns: (num_det_v, num_det_u, num_samples, 3) - det_v × det_u × samples × xyz
```

**Key differences:**
- 2D uses arc detector with single angle parameter
- 3D uses flat-panel detector with u (horizontal) and v (vertical) positions
- 3D adds SDD (Source-to-Detector Distance) parameter
- 3D rays are 3D coordinates (x, y, z) instead of (x, y)

#### Rotation
```python
# 2D: rotate_ray(xy, angle) - 2D rotation matrix
# 3D: rotate_ray_3d(xyz, angle, axis='z') - 3D rotation around x, y, or z axis
```

### 2. **Network (Polyner_3d.py)**

```python
# 2D Network
network = tcnn.NetworkWithInputEncoding(
    n_input_dims=2,  # xy coordinates
    n_output_dims=e_level,
    ...
)

# 3D Network
network = tcnn.NetworkWithInputEncoding(
    n_input_dims=3,  # xyz coordinates
    n_output_dims=e_level,
    ...
)
```

**Everything else remains the same:**
- Hash grid encoding automatically handles 3D
- FullyFusedMLP works with 3D inputs
- Polyenergetic forward model unchanged
- Loss functions (data consistency + ASE) unchanged

### 3. **Data Format**

#### 2D Input Data
```
input/
├── ma_sinogram_0.nii       # (num_angles, num_det)
├── fanSensorPos.nii        # (num_det,) - detector angles
└── mask_0.nii              # (h, w) - metal mask
```

#### 3D Input Data
```
input/
├── ma_projection_0.nii     # (num_angles, num_det_v, num_det_u)
├── detectorUPos.nii        # (num_det_u,) - horizontal detector positions
├── detectorVPos.nii        # (num_det_v,) - vertical detector positions
└── mask_0.nii              # (h, w, d) - 3D metal mask
```

## Usage

### Basic Usage

```bash
cd Polyner
python main_3d.py
```

This will:
1. Load `config_3d.json`
2. Read 3D projection data from `./input/`
3. Train the 3D Polyner model
4. Save reconstructed 3D volumes to `./output/`
5. Save model weights to `./model/`

### Configuration

Edit `config_3d.json` to adjust parameters:

```json
{
  "file": {
    "h": 200,          // Volume height (x dimension)
    "w": 200,          // Volume width (y dimension)
    "d": 150,          // Volume depth (z dimension)
    "SOD": 100,        // Source-to-Origin Distance
    "SDD": 200,        // Source-to-Detector Distance
    "voxel_size": 0.02 // Integration step size
  },
  "train": {
    "epoch": 4000,
    "batch_size": 40,  // Reduce if memory issues (try 20-30)
    "num_sample_ray": 4 // Reduce if memory issues (try 2-3)
  },
  "network": {
    "n_neurons": 128,  // Increase for more capacity (try 256-512)
    "n_hidden_layers": 2 // Increase for more capacity (try 3-4)
  }
}
```

## Memory Optimization

The paper reports **~10 GB** memory usage for a 200×200×150 volume. If you encounter memory issues:

### Reduce Memory Usage
1. **Decrease batch_size**: `40 → 20 or 30`
2. **Decrease num_sample_ray**: `4 → 2 or 3`
3. **Process in sub-regions**: Reconstruct volume in chunks

### Example Low-Memory Config
```json
{
  "train": {
    "batch_size": 20,
    "num_sample_ray": 2
  }
}
```

## Increasing Network Capacity

For better 3D representation (at the cost of more memory/time):

```json
{
  "network": {
    "n_neurons": 256,        // Increased from 128
    "n_hidden_layers": 4     // Increased from 2
  },
  "encoding": {
    "n_levels": 20,          // Increased from 16
    "log2_hashmap_size": 20  // Increased from 19
  }
}
```

## Typical CBCT Parameters

### Medical CBCT
- **SOD**: 300-600 mm
- **SDD**: 500-1200 mm (SDD/SOD ratio: 1.5-2.5)
- **Detector**: 20-40 cm flat panel
- **Projections**: 180-720 over 360° (0.5-2° increments)
- **Volume**: 200-512 voxels per dimension

### Micro-CT (like the paper's mouse scan)
- **SOD**: 50-150 mm
- **SDD**: 100-300 mm
- **Detector**: 5-10 cm flat panel
- **Projections**: 360-720 over 360°
- **Volume**: 200-400 voxels per dimension

## Expected Performance

Based on the paper's results:

| Volume Size | Memory | Time (RTX TITAN) | Projections |
|------------|--------|------------------|-------------|
| 200×200×150 | ~10 GB | ~32 min | 360-720 |

Your performance will vary based on:
- GPU memory and compute capability
- Batch size and num_sample_ray
- Network size (neurons, layers)
- Number of training epochs

## Data Preparation

### From MATLAB Simulation

If you have MATLAB cone-beam projection data:

```matlab
% Generate cone-beam projections using MATLAB
% Save as NIfTI files
niftiwrite(projections, 'ma_projection_0.nii');
niftiwrite(detector_u_positions, 'detectorUPos.nii');
niftiwrite(detector_v_positions, 'detectorVPos.nii');
niftiwrite(metal_mask, 'mask_0.nii');
```

### From Real CBCT Scanner

1. Export projections as 3D array: `(num_angles, num_det_v, num_det_u)`
2. Calculate detector positions in degrees relative to central ray
3. Create metal segmentation mask from initial reconstruction
4. Save all as NIfTI (.nii) files

## Output

After training, you'll find in `./output/`:

```
polyner_3d_0_0.00123_1.23it_s.nii
```

Format: `polyner_3d_{img_id}_{loss}_{iterations_per_sec}it_s.nii`

This is a 3D volume (h×w×d) that can be visualized with:
- 3D Slicer
- ITK-SNAP
- ParaView
- MATLAB
- Python (SimpleITK, nibabel)

## Comparison: 2D vs 3D

| Aspect | 2D Fan-Beam | 3D Cone-Beam |
|--------|-------------|--------------|
| Input dims | 2 (x, y) | 3 (x, y, z) |
| Projections | (angles, detectors) | (angles, det_v, det_u) |
| Reconstruction | 2D slice | 3D volume |
| Memory | Lower (~2-4 GB) | Higher (~10 GB) |
| Time | Faster (~10 min) | Slower (~32 min) |
| Benefit | Faster comparison | Better z-axis consistency |

## Troubleshooting

### Out of Memory
- Reduce `batch_size` (try 20 or 30)
- Reduce `num_sample_ray` (try 2 or 3)
- Reduce volume size temporarily for testing

### Slow Training
- Ensure CUDA is available: `torch.cuda.is_available()`
- Check GPU utilization: `nvidia-smi`
- Reduce `epoch` for testing (try 1000)

### Poor Reconstruction Quality
- Increase network capacity (more neurons/layers)
- Increase hash grid levels
- Train for more epochs
- Check data normalization and preprocessing
- Verify geometry parameters (SOD, SDD) match your scanner

### Artifacts in Reconstruction
- Verify detector positions are correct
- Check metal mask accuracy
- Ensure projections are properly calibrated
- Try adjusting `lambda` (ASE loss weight)

## Citation

If you use this 3D adaptation, please cite the original Polyner paper:

```bibtex
@inproceedings{wu2023polyner,
  title={Polyner: Implicit neural representation for metal artifact reduction},
  author={Wu, Qing and others},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```

## References

1. Original paper: https://openreview.net/pdf?id=xx3QgKyghS
2. Original repository: https://github.com/iwuqing/Polyner
3. Authors' 3D implementation discussion: OpenReview reviewer responses

## Contact

For questions about the original 2D implementation:
- Author: Qing Wu
- Email: wuqing@shanghaitech.edu.cn

For questions about this 3D adaptation:
- Open an issue on the GitHub repository

## License

Same as original Polyner implementation.
