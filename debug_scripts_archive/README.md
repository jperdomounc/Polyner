# Debug Scripts Archive

This directory contains all the debugging and fix scripts that were used to resolve issues with the Polyner data format. These scripts have already been applied to fix the data, but are preserved here for reference.

## 📁 Script Overview:

### **Data Conversion Scripts:**
- **`convert_and_display.py`** - Converted original NRRD files to NII format with visualization
- **`rotate_and_display.py`** - Tested different image rotations to check orientation
- **`resize_unc_data.py`** - Resized images from 216×216 to 256×256 to match repository format

### **Data Format Fix Scripts:**
- **`ensure_all_2d.py`** - Fixed 3D→2D conversion to resolve numpy padding broadcast error
- **`fix_mask_binary.py`** - Converted float mask values to proper binary (0,1) format
- **`normalize_nifti.py`** - Normalized data ranges to physically realistic attenuation coefficients
- **`fix_spectrum_key.py`** - Renamed spectrum dictionary key from 'HE_MAR' to 'GE14Spectrum120KVP'

### **Analysis Scripts:**
- **`compare_inputs.py`** - Compared data formats between user data and original repository

## 🔧 Issues These Scripts Fixed:

1. **Numpy Padding Error**: `ensure_all_2d.py` fixed 3D vs 2D format mismatch
2. **Non-Binary Mask**: `fix_mask_binary.py` converted float values to binary [0,1]
3. **Wrong Dimensions**: `resize_unc_data.py` scaled 216×216→256×256 
4. **Unrealistic Ranges**: `normalize_nifti.py` scaled to proper attenuation coefficients
5. **Spectrum KeyError**: `fix_spectrum_key.py` renamed dictionary key
6. **Format Verification**: `compare_inputs.py` identified all mismatches

## ⚠️ Usage Notes:

- **These scripts have already been applied** - your input data is now fixed
- They are preserved here for **reference and understanding** of what was done
- **Do not run these scripts again** unless you want to revert and re-apply fixes
- The fixes are **permanent** - your input folder data has been corrected

## 🎯 Current Status:

All issues identified by these scripts have been resolved:
- ✅ Binary mask format
- ✅ Proper 2D dimensions  
- ✅ Realistic value ranges
- ✅ Correct spectrum key
- ✅ Matching repository format
- ✅ Fixed config parameters

Your data is now ready for successful Polyner training!