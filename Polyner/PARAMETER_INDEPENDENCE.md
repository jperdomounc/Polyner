# Parameter Independence: SOD, SDD, and num_samples

## Summary of Changes

The 3D cone-beam implementation has been updated to ensure **complete independence** between the physical geometry parameters (SOD, SDD) and the computational parameters (num_samples, volume dimensions).

## Parameters Explained

### Physical Geometry Parameters (Scanner-Specific)

1. **SOD (Source-to-Origin Distance)**
   - Physical distance from X-ray source to isocenter
   - Measured in mm
   - Example: 100 mm (micro-CT), 500 mm (medical CBCT)
   - **Purpose**: Defines cone-beam geometry and magnification

2. **SDD (Source-to-Detector Distance)**
   - Physical distance from X-ray source to detector panel
   - Measured in mm
   - Example: 200 mm (micro-CT), 1000 mm (medical CBCT)
   - **Purpose**: Determines detector placement and cone angle
   - **Independent from SOD**: Typical ratio SDD/SOD = 1.5 to 2.5

### Computational Parameters (Reconstruction-Specific)

3. **num_samples**
   - Number of discrete samples along each ray
   - No units (integer count)
   - Example: 200 samples
   - **Purpose**: Ray discretization for numerical integration
   - **Independent from SOD and SDD**: Higher = more accurate but more memory

4. **h, w, d (Volume Dimensions)**
   - Size of reconstruction volume in voxels
   - Example: 200×200×150 voxels
   - **Purpose**: Defines output image resolution
   - **Independent from SOD, SDD, num_samples**

## Before vs After

### ❌ Before (Incorrect - Coupled Parameters)

```python
# In utils_3d.py - WRONG
def cone_beam_ray(detector_u_pos, detector_v_pos, SOD, SDD):
    num_samples = int(2 * SOD)  # ❌ Tied to SOD!
    # ...

# In Polyner_3d.py - WRONG
test_loader = data.DataLoader(
    dataset=TestData3D(h=(2*SOD)+1, w=(2*SOD)+1, d=(2*SOD)+1),  # ❌ Tied to SOD!
    # ...
)
```

**Problems:**
- Volume size forced to be `(2*SOD+1)³` regardless of actual needs
- Can't independently control ray sampling resolution
- Confuses physical parameters with computational parameters

### ✅ After (Correct - Independent Parameters)

```python
# In utils_3d.py - CORRECT
def cone_beam_ray(detector_u_pos, detector_v_pos, SOD, SDD, num_samples):
    # num_samples is now an explicit parameter ✓
    # SOD and SDD only used for geometry ✓
    # ...

# In Polyner_3d.py - CORRECT
test_loader = data.DataLoader(
    dataset=TestData3D(h=h, w=w, d=d),  # ✓ Uses actual volume dimensions
    # ...
)
```

**Benefits:**
- Physical scanner geometry (SOD, SDD) independent from reconstruction parameters
- Can reconstruct any volume size (e.g., 200×200×150) with any scanner geometry
- Can tune `num_samples` for accuracy/memory tradeoff independently

## Configuration File

### config_3d.json

```json
{
  "file": {
    "SOD": 100,          // Physical: Source-to-Origin Distance (mm)
    "SDD": 200,          // Physical: Source-to-Detector Distance (mm)
    "num_samples": 200,  // Computational: Ray discretization
    "h": 200,            // Computational: Volume width (voxels)
    "w": 200,            // Computational: Volume height (voxels)
    "d": 150             // Computational: Volume depth (voxels)
  }
}
```

### Parameter Relationships

```
Physical Geometry (scanner hardware):
  SOD, SDD → Define cone-beam angles and magnification
  └─ Typical: SDD/SOD ≈ 1.5-2.5

Computational (algorithm parameters):
  num_samples → Ray discretization (integration accuracy)
  h, w, d → Output volume resolution
  └─ All independent from SOD, SDD!
```

## Example Configurations

### Example 1: Micro-CT (Small Scanner)
```json
{
  "SOD": 100,          // Small isocenter distance
  "SDD": 200,          // Detector close to source
  "num_samples": 300,  // High sampling for accuracy
  "h": 512, "w": 512, "d": 512  // High resolution volume
}
```

### Example 2: Medical CBCT (Large Scanner)
```json
{
  "SOD": 500,          // Large isocenter distance
  "SDD": 1000,         // Detector far from source
  "num_samples": 200,  // Moderate sampling
  "h": 256, "w": 256, "d": 200  // Moderate resolution
}
```

### Example 3: Same Scanner, Different Reconstruction
```json
// Configuration A: High quality, small ROI
{
  "SOD": 100,          // Same scanner
  "SDD": 200,          // Same scanner
  "num_samples": 400,  // More samples for accuracy
  "h": 128, "w": 128, "d": 128  // Small, high-quality ROI
}

// Configuration B: Lower quality, large volume
{
  "SOD": 100,          // Same scanner
  "SDD": 200,          // Same scanner
  "num_samples": 150,  // Fewer samples for speed
  "h": 512, "w": 512, "d": 384  // Large volume
}
```

## How to Choose Parameters

### SOD and SDD (Match Your Scanner)
```
1. Check your scanner's technical specifications
2. SOD = source-to-isocenter distance
3. SDD = source-to-detector distance
4. These are FIXED by your hardware
```

### num_samples (Tune for Accuracy vs Memory)
```
Higher values:
  ✓ More accurate integration
  ✓ Better reconstruction quality
  ✗ More memory usage
  ✗ Slower computation

Typical values:
  - 100-200: Fast, lower quality
  - 200-300: Balanced (recommended)
  - 300-500: High quality, slow
```

### h, w, d (Choose Based on Application)
```
Factors to consider:
  1. Actual object size you want to reconstruct
  2. Desired voxel size (physical resolution)
  3. Available memory
  4. Computation time

Example calculation:
  Object size: 40mm × 40mm × 30mm
  Desired voxel size: 0.2mm
  Volume dimensions: 200 × 200 × 150 voxels
```

## Memory Usage

Memory usage scales with:
```
Memory ∝ num_samples × batch_size × num_sample_ray × energy_levels

Example:
  num_samples = 200
  batch_size = 40
  num_sample_ray = 4
  energy_levels = 101

  Memory per forward pass ≈ 200 × 40 × 4 × 101 × 4 bytes ≈ 13 MB
  Plus network parameters, gradients, etc. → ~10 GB total
```

If you encounter memory issues:
1. **Reduce batch_size**: 40 → 20 or 30
2. **Reduce num_sample_ray**: 4 → 2 or 3
3. **Reduce num_samples**: 200 → 150 (less impact on quality)

## Technical Details

### Ray Generation (utils_3d.py)

```python
def cone_beam_ray(detector_u_pos, detector_v_pos, SOD, SDD, num_samples):
    """
    Generate 3D rays for cone-beam geometry.

    SOD and SDD: Define the physical geometry
    num_samples: Controls discretization (independent)
    """
    # Detector position calculation uses SOD and SDD
    det_distance_normalized = SDD / SOD  # Geometry ratio
    det_x = det_distance_normalized * np.tan(cone_angle_u)
    det_y = det_distance_normalized - 1
    det_z = det_distance_normalized * np.tan(cone_angle_v)

    # Ray sampling uses num_samples (independent)
    t = np.linspace(0, 2, num_samples)
    ray = source + t * (detector - source)
```

### Volume Reconstruction (Polyner_3d.py)

```python
# Volume dimensions completely independent from SOD
test_loader = data.DataLoader(
    dataset=TestData3D(h=h, w=w, d=d),  # From config directly
    ...
)

# Reconstruction outputs h×w×d volume
img_pre = network(xyz)[:, energy_idx].view(h, w, d)
```

## Verification

To verify parameter independence, try these tests:

### Test 1: Same Scanner, Different Resolutions
```bash
# Config 1: 128³ volume
python main_3d.py  # Should work with SOD=100, SDD=200

# Config 2: 256³ volume (edit config_3d.json)
python main_3d.py  # Should work with same SOD=100, SDD=200
```

### Test 2: Different num_samples
```bash
# Try num_samples = 100, 200, 400
# All should work with same SOD, SDD, h, w, d
```

### Test 3: Different Scanner Geometries
```bash
# Micro-CT: SOD=100, SDD=200
# Medical CBCT: SOD=500, SDD=1000
# Both with same h=200, w=200, d=150
```

## Summary

✅ **Correct approach**: SOD, SDD, num_samples, and (h,w,d) are all **independent parameters**

- **SOD, SDD**: Physical scanner geometry
- **num_samples**: Computational ray discretization
- **h, w, d**: Output volume dimensions

You can now freely adjust each parameter based on your specific needs without artificial constraints!
