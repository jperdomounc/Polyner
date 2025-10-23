# Quick Start Guide: 3D Cone-Beam CT with Polyner

## Setup (5 minutes)

### 1. Verify Dependencies
```bash
cd Polyner
python -c "import torch; import tinycudann; import SimpleITK; print('All dependencies OK!')"
```

If any imports fail, install:
```bash
pip install torch torchvision
pip install tiny-cuda-nn
pip install SimpleITK scipy scikit-image tqdm
```

### 2. Prepare Directory Structure
```bash
mkdir -p input output model
```

### 3. Check Your Data

You need these files in `./input/`:

| File | Shape | Description |
|------|-------|-------------|
| `ma_projection_0.nii` | (num_angles, num_det_v, num_det_u) | 3D cone-beam projections |
| `detectorUPos.nii` | (num_det_u,) | Horizontal detector positions (degrees) |
| `detectorVPos.nii` | (num_det_v,) | Vertical detector positions (degrees) |
| `mask_0.nii` | (h, w, d) | 3D metal segmentation mask |
| `GE14Spectrum120KVP.mat` | (120, 2) | X-ray energy spectrum |

## Run (1 command)

```bash
python main_3d.py
```

That's it! The reconstruction will start.

## Monitor Progress

You'll see output like:
```
3D Image #0: 100%|████████| 4000/4000 [32:15<00:00, 2.07it/s, lr=0.001, loss=0.0234]
```

Every 500 epochs, a 3D volume is saved to `./output/`:
```
output/polyner_3d_0_0.00234_1.23it_s.nii
```

## First Time? Start Small!

Edit `config_3d.json` for a quick test run:

```json
{
  "file": {
    "h": 100,
    "w": 100,
    "d": 75
  },
  "train": {
    "epoch": 1000,
    "save_epoch": 250,
    "batch_size": 20
  }
}
```

This will:
- Use smaller volume (100×100×75 instead of 200×200×150)
- Train for 1000 epochs (~8 minutes instead of 32)
- Save every 250 epochs to see progress faster
- Use less memory (20 batch size instead of 40)

## Expected Timeline

| Stage | Time | Output |
|-------|------|--------|
| Data loading | 10-30s | Prints data shapes |
| Epoch 250 | ~2 min | First reconstruction saved |
| Epoch 500 | ~4 min | Second reconstruction |
| Epoch 1000 | ~8 min | Final reconstruction (quick test) |
| Epoch 4000 | ~32 min | Full reconstruction (best quality) |

## Check Results

### Using Python
```python
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# Load 3D volume
img = sitk.GetArrayFromImage(sitk.ReadImage('./output/polyner_3d_0_0.00234_1.23it_s.nii'))

# View middle slices
plt.figure(figsize=(15, 5))
plt.subplot(131); plt.imshow(img[img.shape[0]//2, :, :], cmap='gray'); plt.title('Axial')
plt.subplot(132); plt.imshow(img[:, img.shape[1]//2, :], cmap='gray'); plt.title('Coronal')
plt.subplot(133); plt.imshow(img[:, :, img.shape[2]//2], cmap='gray'); plt.title('Sagittal')
plt.tight_layout()
plt.savefig('reconstruction_slices.png')
```

### Using 3D Slicer (recommended)
1. Download: https://www.slicer.org/
2. File → Add Data → Select `.nii` file
3. Adjust window/level for visualization

## Troubleshooting

### "Out of memory"
```json
{
  "train": {
    "batch_size": 10,
    "num_sample_ray": 2
  }
}
```

### "CUDA not available"
Check: `python -c "import torch; print(torch.cuda.is_available())"`

If False:
- Install CUDA-enabled PyTorch: https://pytorch.org/
- Check NVIDIA drivers: `nvidia-smi`

### "File not found"
Verify your files:
```bash
ls -lh input/
# Should show: ma_projection_0.nii, detectorUPos.nii, detectorVPos.nii, mask_0.nii
```

### "Poor reconstruction quality"
1. Train longer: `"epoch": 8000` or `16000`
2. Larger network: `"n_neurons": 256`, `"n_hidden_layers": 4`
3. Check your input data quality and geometry parameters

## What's Different from 2D?

| | 2D Fan-Beam | 3D Cone-Beam |
|--|-------------|--------------|
| **Script** | `python main.py` | `python main_3d.py` |
| **Config** | `config.json` | `config_3d.json` |
| **Input** | 2D sinogram | 3D projections |
| **Output** | 2D slice | 3D volume |
| **Memory** | ~2-4 GB | ~10 GB |
| **Time** | ~10 min | ~32 min |

## Next Steps

1. **Adjust parameters** in `config_3d.json` for your specific scanner geometry
2. **Visualize** results in 3D Slicer or ParaView
3. **Evaluate** reconstruction quality (PSNR, SSIM) if you have ground truth
4. **Compare** with 2D slice-by-slice reconstruction

## Need Help?

1. Check `README_3D.md` for detailed documentation
2. Review original paper: https://openreview.net/pdf?id=xx3QgKyghS
3. Open an issue on GitHub

## Advanced: Custom Data Preparation

If you have raw CBCT data:

```python
import numpy as np
import SimpleITK as sitk

# Your cone-beam projections: (num_angles, height, width)
projections = ...  # Load from your scanner format

# Detector geometry
num_det_u = projections.shape[2]  # Horizontal pixels
num_det_v = projections.shape[1]  # Vertical pixels

# Detector angular positions (adjust based on your geometry)
pixel_size = 0.2  # mm
SOD = 500  # mm - your scanner's source-to-isocenter distance

u_positions = np.arctan(np.arange(num_det_u) - num_det_u/2) * pixel_size / SOD * 180/np.pi
v_positions = np.arctan(np.arange(num_det_v) - num_det_v/2) * pixel_size / SOD * 180/np.pi

# Save as NIfTI
sitk.WriteImage(sitk.GetImageFromArray(projections), 'input/ma_projection_0.nii')
sitk.WriteImage(sitk.GetImageFromArray(u_positions), 'input/detectorUPos.nii')
sitk.WriteImage(sitk.GetImageFromArray(v_positions), 'input/detectorVPos.nii')

# Create metal mask (you'll need to segment this)
# Option 1: Threshold initial reconstruction
# Option 2: Use metal thresholding (e.g., >3000 HU)
mask = ...  # Your 3D metal segmentation
sitk.WriteImage(sitk.GetImageFromArray(mask.astype(np.uint8)), 'input/mask_0.nii')
```

Good luck with your 3D cone-beam reconstruction!
