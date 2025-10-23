# 3D Cone-Beam CT Adaptation - Technical Changes Summary

## Overview

This document summarizes the exact changes made to adapt Polyner from 2D fan-beam CT to 3D cone-beam CT, following the authors' guidance that "only a simple modification following 3D acquisition geometry for X-ray simulation needs to be conducted, while all other steps remain unchanged."

## File Structure

### New 3D Files Created
```
Polyner/
├── utils_3d.py          # 3D geometry functions
├── dataset_3d.py        # 3D data loaders
├── Polyner_3d.py        # 3D training loop
├── config_3d.json       # 3D configuration
├── main_3d.py           # 3D entry point
├── README_3D.md         # Comprehensive documentation
├── QUICKSTART_3D.md     # Quick start guide
└── CHANGES_3D.md        # This file
```

### Original 2D Files (Unchanged)
```
Polyner/
├── utils.py             # Original 2D geometry
├── dataset.py           # Original 2D loaders
├── Polyner.py           # Original 2D training
├── config.json          # Original 2D config
├── main.py              # Original 2D entry
├── model.py             # Shared loss functions (no changes needed)
└── eval.py              # Evaluation metrics (works for both 2D/3D)
```

## Key Changes

### 1. X-Ray Geometry Simulation (utils_3d.py)

This is the **main change** as mentioned by the authors.

#### 2D Fan-Beam Ray Generation
```python
def fan_beam_ray(proj_pos, SOD):
    """
    2D fan-beam geometry with arc detector

    Input:
        proj_pos: (num_det,) - detector angles in degrees
        SOD: scalar - source to origin distance

    Output:
        rays: (num_det, 2*SOD, 2) - detector × samples × (x,y)

    Geometry:
        - Source at (0, -1) in normalized coordinates
        - Arc detector with angular positions
        - 2D rotation matrix for fan angles
    """
```

#### 3D Cone-Beam Ray Generation
```python
def cone_beam_ray(detector_u_pos, detector_v_pos, SOD, SDD):
    """
    3D cone-beam geometry with flat-panel detector

    Input:
        detector_u_pos: (num_det_u,) - horizontal detector angles
        detector_v_pos: (num_det_v,) - vertical detector angles
        SOD: scalar - source to origin distance
        SDD: scalar - source to detector distance

    Output:
        rays: (num_det_v, num_det_u, num_samples, 3) - det_v × det_u × samples × (x,y,z)

    Geometry:
        - Source at (0, -1, 0) in normalized coordinates
        - Flat-panel detector at distance SDD
        - 3D ray casting from source through detector to volume
    """
```

**Mathematical differences:**

2D Arc Detector:
```
detector_position = [cos(θ), sin(θ)] * radius
ray = source + t * (detector - source)
```

3D Flat Panel:
```
detector_position = [tan(θ_u) * d, d, tan(θ_v) * d]
ray = source + t * (detector - source)
where d = SDD/SOD (normalized)
```

#### Rotation Functions

2D:
```python
def rotate_ray(xy, angle):
    """2D rotation matrix around origin"""
    R = [[cos(θ), -sin(θ)],
         [sin(θ),  cos(θ)]]
```

3D:
```python
def rotate_ray_3d(xyz, angle, axis='z'):
    """3D rotation around specified axis (x, y, or z)"""
    # Z-axis rotation (typical CT gantry):
    R = [[cos(θ), -sin(θ), 0],
         [sin(θ),  cos(θ), 0],
         [0,       0,      1]]
```

### 2. Network Input Dimensions (Polyner_3d.py)

**The only change in the network itself:**

2D:
```python
network = tcnn.NetworkWithInputEncoding(
    n_input_dims=2,  # Input: (x, y) coordinates
    n_output_dims=e_level,
    encoding_config=config["encoding"],
    network_config=config["network"]
)
```

3D:
```python
network = tcnn.NetworkWithInputEncoding(
    n_input_dims=3,  # Input: (x, y, z) coordinates
    n_output_dims=e_level,
    encoding_config=config["encoding"],
    network_config=config["network"]
)
```

**Everything else in the network architecture remains identical:**
- Hash grid encoding (automatically adapts to 3D)
- FullyFusedMLP architecture
- Output dimensions (energy levels)
- Activation functions

### 3. Data Loaders (dataset_3d.py)

#### 2D Training Data
```python
class TrainData(data.Dataset):
    # Loads sinogram: (num_angles, num_det)
    # Generates rays: (num_det, 2*SOD, 2)
    # Returns: ray (num_sample_ray, 2*SOD, 2), proj (num_sample_ray,)
```

#### 3D Training Data
```python
class TrainData3D(data.Dataset):
    # Loads projections: (num_angles, num_det_v, num_det_u)
    # Generates rays: (num_det_v, num_det_u, num_samples, 3)
    # Returns: ray (num_sample_ray, num_samples, 3), proj (num_sample_ray,)
```

**Key differences:**
- 3D requires two detector position files (u and v)
- 3D projection data has extra dimension (vertical detector)
- Ray sampling adapted for 2D detector array

### 4. Configuration (config_3d.json)

#### New 3D Parameters
```json
{
  "file": {
    "d": 150,              // NEW: Volume depth (z dimension)
    "SDD": 200,            // NEW: Source-to-Detector Distance
    "SOD": 100,            // Changed from 2050 (different scale for mouse CT)
    "h": 200, "w": 200     // Similar to 2D but for 3D volume
  }
}
```

#### Unchanged Parameters
```json
{
  "train": {
    "lr": 1e-3,
    "epoch": 4000,
    "batch_size": 40,      // May need reduction for memory
    "num_sample_ray": 4,
    "lambda": 0.2
  },
  "encoding": { ... },     // Hash grid - same config
  "network": { ... }       // MLP architecture - same config
}
```

### 5. Forward Model (Unchanged!)

**This is crucial - the forward model requires NO changes:**

```python
# 2D and 3D use IDENTICAL forward model:

# Line integral along ray
proj_pre = torch.exp(-voxel_size * torch.sum(intensity_pre, dim=2))

# Polyenergetic spectrum integration
proj_pre = -torch.log(torch.sum(proj_pre * spectrum, dim=-1))
```

**Why it works:**
- Integration is always along the ray (dim=2 in both cases)
- Input is (batch, num_rays, num_samples, energy_levels)
- Output is (batch, num_rays) in both 2D and 3D
- Only difference: samples are 2D (xy) vs 3D (xyz) coordinates

### 6. Loss Functions (model.py - No Changes!)

Both losses work identically for 2D and 3D:

```python
# Data Consistency Loss
L_DC = L1(proj_predicted, proj_measured)

# Attenuation Smoothing over Energies (ASE)
L_ASE = smoothness_over_energy_dimension(attenuation_map)

# Total Loss
L = L_DC + λ * L_ASE
```

The ASE loss operates on the energy dimension, independent of spatial dimensions.

## What Remains Unchanged

As the authors stated, "all other steps remain unchanged":

✅ **Network Architecture**
- Hash grid encoding parameters
- MLP layer configuration
- Activation functions
- Output dimensions (energy levels)

✅ **Forward Model**
- Beer's Law: I = I₀ * exp(-∫μ dl)
- Line integral computation
- Polyenergetic spectrum integration
- Log projection transformation

✅ **Loss Functions**
- Data consistency (L1 loss)
- ASE regularization
- Loss weighting (λ)

✅ **Optimization**
- Adam optimizer
- Learning rate schedule
- Training loop structure

✅ **Energy Representation**
- 101 energy levels (20-120 keV)
- GE spectrum integration
- Energy-dependent attenuation

## Performance Comparison

| Aspect | 2D Fan-Beam | 3D Cone-Beam | Change |
|--------|-------------|--------------|--------|
| Network input | (x, y) | (x, y, z) | +1 dimension |
| Detector array | 1D arc | 2D flat panel | +1 dimension |
| Reconstruction | 2D slice | 3D volume | +1 dimension |
| Memory usage | ~2-4 GB | ~10 GB | ~3-5× increase |
| Training time | ~10 min | ~32 min | ~3× increase |
| Ray samples | 2*SOD | 2*SOD | Same |
| Energy levels | 101 | 101 | Same |
| Network layers | 2 hidden | 2 hidden | Same |
| Hash levels | 16 | 16 | Same (can increase) |

## Implementation Checklist

To adapt any 2D CT INR model to 3D cone-beam:

- [x] **X-ray geometry**: Implement cone-beam ray generation
- [x] **Network input**: Change from 2D to 3D coordinates
- [x] **Data loader**: Handle 3D projection data
- [x] **Configuration**: Add z-dimension and SDD parameters
- [ ] **Forward model**: No changes needed! ✓
- [ ] **Loss functions**: No changes needed! ✓
- [ ] **Optimization**: No changes needed! ✓

## Code Diff Summary

### Critical Changes (Geometry)
```diff
- def fan_beam_ray(proj_pos, SOD):
+ def cone_beam_ray(detector_u_pos, detector_v_pos, SOD, SDD):

- xy = np.zeros(shape=(num_det, int(2*SOD), 2))
+ rays = np.zeros((num_det_v, num_det_u, num_samples, 3))

- det_x = ...  # 2D calculation
+ det_x, det_y, det_z = ...  # 3D calculation
```

### Critical Changes (Network)
```diff
- network = tcnn.NetworkWithInputEncoding(n_input_dims=2, ...)
+ network = tcnn.NetworkWithInputEncoding(n_input_dims=3, ...)

- ray.view(-1, 2)
+ ray.view(-1, 3)

- grid_coordinate(h, w)
+ grid_coordinate_3d(h, w, d)
```

### No Changes Needed
```python
# Forward model - IDENTICAL in both versions
proj_pre = torch.exp(-voxel_size * torch.sum(intensity_pre, dim=2))
proj_pre = -torch.log(torch.sum(proj_pre * spectrum, dim=-1))

# Loss - IDENTICAL
loss = dc_loss(proj_pre, proj) + ase_loss(intensity=intensity_pre, ray=ray)
```

## Validation

To verify the 3D implementation matches the 2D approach:

1. **Geometry Test**: Check that rays from source through detector sample the volume correctly
2. **Projection Test**: Verify forward projection matches scanner geometry
3. **Reconstruction Test**: Ensure volume converges to artifact-reduced solution
4. **Memory Test**: Confirm ~10 GB usage for 200×200×150 volume (as reported in paper)
5. **Time Test**: Expect ~32 minutes on RTX TITAN for 4000 epochs

## References

1. **Original Paper**: Wu et al., "Polyner: Implicit Neural Representation for Metal Artifact Reduction", ICLR 2023
2. **3D Discussion**: OpenReview reviewer response (Figure R1 showing mouse CT scan)
3. **Geometry**: Standard cone-beam CT geometry from Feldkamp et al., 1984

## Notes

- The elegance of the INR approach is that the forward model is **dimension-agnostic**
- Hash grid encoding automatically scales to 3D without parameter changes
- Main engineering effort: proper 3D ray generation for cone-beam geometry
- Network capacity may need increase for larger 3D volumes (more neurons/layers)

---

**Summary**: The adaptation from 2D to 3D is remarkably simple, requiring only geometry changes and network input dimension changes. The core INR methodology, forward model, and loss functions transfer directly to 3D with no modifications.
