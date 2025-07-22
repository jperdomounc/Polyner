# Polyner Debugging Suite - Instructions

## 🚨 CRITICAL ISSUES FOUND & FIXED

### Issue 1: Non-Binary Mask (FIXED ✓)
- **Problem**: Your mask contained float values (-0.179 to 1.227) instead of binary (0,1)
- **Impact**: Algorithm couldn't identify metal regions (0% metal pixels)
- **Fix Applied**: Converted to proper binary mask with 0.2% metal pixels (133 pixels)

### Issue 2: Spectrum Key Mismatch (FIXED ✓)  
- **Problem**: Spectrum file had key 'HE_MAR' instead of 'GE14Spectrum120KVP'
- **Impact**: KeyError when loading spectrum
- **Fix Applied**: Renamed key while preserving your spectrum data

## 📊 Current Data Status

### ✅ WORKING:
- **Data Dimensions**: All files properly sized (256×256 images, 360×611 sinogram)
- **Spectrum**: 150 energy points, 1-150 keV, proper intensity distribution
- **File Loading**: All input files load correctly
- **Mask**: Now properly binary (0.2% metal, 99.8% background)

### ⚠️ RECOMMENDATIONS:
1. **Metal Percentage**: 0.2% is quite low - typical metal artifacts are 2-10%
2. **Consider**: If results are still poor, you may need to adjust mask threshold or check if metal regions are correctly identified

## 🔧 Debugging Tools Created

### 1. Main Debugging Suite (`debug_polyner.py`)
```python
from debug_polyner import PolynerDebugger

debugger = PolynerDebugger()
results = debugger.run_full_analysis(img_id=0)
```

**What it does:**
- ✅ Checks data consistency (dimensions, ranges, types)
- ✅ Validates mask (binary check, metal detection, padding)
- ✅ Verifies spectrum (energy range, intensity, format)
- ✅ Creates visualizations and plots
- ✅ Generates debug config

### 2. Iteration Monitor (`polyner_debug_patch.py`)
For monitoring training iterations and p-values:

```python
# Add to your training script:
from polyner_debug_patch import init_monitor, log_iteration, finalize_monitoring

# Initialize
monitor = init_monitor("./debug_output")

# In your training loop (add this around line 50 in Polyner.py):
log_iteration(epoch, loss, p_hat, proj_pre, intensity_pre)

# After training:
finalize_monitoring()
```

**What it monitors:**
- 📈 P-value statistics at each iteration (mean, std, min, max)
- 💾 Projection and intensity data storage
- 📊 Loss curves and convergence
- 📁 Saves all data for post-analysis

### 3. Mask Binary Fix (`fix_mask_binary.py`)
- ✅ Already applied to your data
- Tests different thresholds
- Converts float masks to proper binary format

## 🎯 Next Steps for Colab

### 1. Upload Fixed Data
Upload your corrected input folder to Colab with:
- ✅ Binary mask (`mask_0.nii`)  
- ✅ Proper spectrum key (`GE14Spectrum120KVP.mat`)
- ✅ All files in 256×256 format

### 2. Add Debug Monitoring
In your Colab notebook, add this to monitor training:

```python
# Upload polyner_debug_patch.py to Colab
from polyner_debug_patch import init_monitor, log_iteration, finalize_monitoring

# Before training
monitor = init_monitor()

# Modify your Polyner.py training loop to include:
# Around line 50, after computing p_hat:
log_iteration(epoch, loss.item(), p_hat, proj_pre, intensity_pre)

# After training completes:
finalize_monitoring()
```

### 3. Monitor Key Values
Watch for these indicators during training:

**P-Values (Line 50 monitoring):**
- Should converge to reasonable range (not exploding)
- Mean should stabilize after initial epochs
- Std should decrease over time

**Loss:**
- Should decrease consistently
- No sudden jumps or NaN values

**Projections/Intensity:**
- Should remain in physically reasonable ranges
- No infinite or NaN values

## 📋 Debug Checklist

Before running in Colab:
- [ ] ✅ Mask is binary (0s and 1s only)
- [ ] ✅ Spectrum has correct key 'GE14Spectrum120KVP'  
- [ ] ✅ All images are 256×256
- [ ] ✅ Sinogram is 360×611
- [ ] ✅ Config parameters match your data (SOD=410, voxel_size=1)
- [ ] 📊 Debug monitoring added to training loop

## 🔍 Understanding Line 50-58 Issues

### Line 50: P-value monitoring
- `p_hat` represents predicted projection values
- Should be physically meaningful (not negative, not infinite)
- Monitor convergence behavior

### Line 51: Mask padding/rotation
- ✅ Fixed: Mask now properly binary
- Padding adds background around mask to match SOD geometry
- Rotation aligns with coordinate system

### Line 58: Spectrum loading
- ✅ Fixed: Correct key name
- Spectrum defines X-ray energy distribution
- Critical for polychromatic reconstruction

## 📊 Expected Results After Fixes
With the mask and spectrum fixes, you should see:
- ✅ No more KeyError crashes
- ✅ No more "no metal pixels" issues  
- 📈 Better reconstruction quality
- 📉 More stable training convergence

## 🆘 If Problems Persist

1. **Check metal percentage**: If 0.2% is too low, try lower threshold (0.3 or 0.1)
2. **Monitor p-values**: Look for NaN, infinity, or unrealistic ranges
3. **Verify geometry**: Ensure SOD, image size, and sinogram dimensions are consistent
4. **Compare with original repository data**: Use files from `input_compare/` as reference

Your data is now properly formatted and should work much better with the Polyner algorithm!