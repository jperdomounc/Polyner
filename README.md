# Polyner for the UNC Multisource Array CBCT System

Metal artifact reduction (MAR) for the University of North Carolina multisource array cone-beam CT system, adapted from the Polyner method in the NeurIPS 2023 paper *Unsupervised Polychromatic Neural Representation for CT Metal Artifact Reduction* by Wu et al. ([OpenReview](https://openreview.net/forum?id=xx3QgKyghS), [arXiv](https://arxiv.org/abs/2306.15203)).

Polyner fits an implicit neural representation (a hash-encoded MLP) to a single scan's projection data. The network maps a 3D coordinate to a vector of linear attenuation coefficients, one per energy bin, and is trained by pushing those coefficients through a polychromatic forward model and matching the measured line integrals. By modeling metal as energy-dependent attenuation rather than treating its measurements as missing data, the method aims to suppress beam-hardening streaks without discarding the metal trace. There is no training set and no pretrained weights: each scan gets its own optimization run.

This fork replaces the public reference code's 2D fan-beam pipeline with UNC's 3D cone-beam setup, where detector element positions come from measured fan/cone angle files rather than an idealized arc. The paper itself also reports a separate 3D cone-beam mouse-thigh experiment.

> [!IMPORTANT]
> This is research code, not a clinically validated reconstruction system. Inspect geometry, orientation, units, and metal-mask alignment for every dataset before interpreting an output.

## Research history and current status

This repository combines two kinds of evidence. The Wu et al. paper defines the original Polyner method: a case-specific implicit neural representation, a differentiable polychromatic forward model, and an energy-dependent smoothness (EDS) loss outside metal. The project PowerPoints record the practical UNC adaptation and experiments. They should not be read as new instructions from the paper.

- In 2025, the original 2D fan-beam results were reproduced and dependencies were tuned in Google Colab. The project notes report roughly 7--10 minutes for a 4,000-epoch 2D run on Colab T4/L4 GPUs; runtime depends heavily on the GPU and volume size.
- The code then moved from fan-beam slices to 3D cone-beam geometry, UNC detector-angle files, RANDO phantom data, and NIfTI volumes. A 1 mm, `200 x 200 x 64` experiment reached 40,000 epochs; the notes indicate that image quality changed little after about 10,000 epochs for that run.
- The March--April 2026 work added uniform random detector-ray sampling, dense-view reprojection, metal-as-air reprojection, and conventional ASTRA reconstruction. The consolidation used here starts from `working-cbct2` at commit `6dcab93`, the newest implementation line after comparing and refreshing every remote branch.
- The documented best hyperparameter family used 16 sampled rays per angle, a batch of 40 angles, learning-rate decay `0.5` every 4,000 epochs, and two hidden layers. A 64-neuron network reached essentially the same loss as 128 neurons; deeper variants sometimes diverged to NaN. The checked-in config keeps the paper-style 2 x 128 network.

Loss values in the slides are experiment logs, not directly comparable benchmarks: the sampling strategy, ray count, volume, and effective number of full-data cycles changed between runs. The paper's PSNR/SSIM results also describe its own datasets, not this UNC branch.

### Relationship to the VFINAF project paper

The later project manuscript, *Volumetric Fan-Beam Imaging with Neural Attenuation Field for Cone-Beam Artifact Reduction*, describes the next research direction built from this work. VFINAF learns a scalar 3D neural attenuation field from measured cone-beam data, then queries it slice by slice along 1,440 virtual fan-beam views and reconstructs each axial slice with conventional FBP or SIRT. The goal is cone-beam artifact reduction by decoupling the final fan-beam reconstruction geometry from the incomplete circular cone-beam acquisition geometry.

This branch is related research code, not a complete implementation of that manuscript. In particular, `reprojection.py` currently generates a 3D cone-beam sinogram and `astra_recon.py` performs a 3D cone-beam reconstruction; the manuscript's defining second stage instead generates a separate 2D fan-beam sinogram for every axial slice. The branch also retains Polyner's polychromatic, multi-output metal-artifact model, whereas the VFINAF manuscript describes a scalar attenuation field and an L1 projection loss.

[ASTRA Toolbox](https://astra-toolbox.com/) was used for conventional FDK/SIRT reconstruction and the downstream reconstruction experiments. ASTRA supports flexible 2D fan-beam and 3D cone-beam source/detector geometry, including the `cone_vec` representation used by `astra_recon.py`; see its [3D geometry documentation](https://astra-toolbox.com/docs/geom3d.html), [FDK CUDA documentation](https://astra-toolbox.com/docs/algs/FDK_CUDA.html), and [SIRT3D CUDA documentation](https://astra-toolbox.com/docs/algs/SIRT3D_CUDA.html).

### Why Google Colab and MATLAB appear in the workflow

Google Colab was used for CUDA GPU access and for resolving the PyTorch/tiny-cuda-nn dependency stack before longer cone-beam runs. MATLAB was used for parts of the image-reconstruction and data-preparation workflow, including work around fan-beam tools, projection/reconstruction experiments, `.mat` spectra, and NIfTI interchange. Those MATLAB-side steps are not all present as scripts on this branch, so the repository starts from the prepared `.mat` and `.nii` inputs.

The `matnifti.py` utility makes that boundary explicit. MATLAB writes NIfTI arrays in column-major order without the axis reversal introduced by `SimpleITK.GetImageFromArray`; `matnifti.niftiwrite`, `niftiread`, and `niftiinfo` reproduce MATLAB's indexing and header conventions. `test_matnifti.py` checks byte-identical round trips against the included MATLAB-created inputs as well as NIfTI-1/2, compressed, paired-file, endian, transform, and scaling cases.

The MATLAB script now included at `matlab/generate_metal_mask_from_sirt.m` documents one concrete preprocessing step: generating a binary metal mask from a SIRT reconstruction. It uses MATLAB's `niftiread`, `niftiinfo`, and `niftiwrite` functions plus Image Processing Toolbox morphology. The default workflow thresholds the top 0.5% of intensities, fills holes slice by slice, removes small 3D components, closes and dilates the mask, preserves the source NIfTI metadata, and writes both dilated and tight masks. Thresholds and morphology must be inspected for each volume; the example manual threshold of 3000 is meaningful for HU-like data, not necessarily for linear attenuation coefficients.

---

## 1. Pipeline overview

The project is three sequential stages, each a standalone script run from the repository root.

```
input/<dataset>/LE/                      config.json
  proj_*.nii  (measured sinogram)   ┐        │
  fanSensorPosition_fanangle*.nii   │        │
  fanSensorPosition_coneangle*.nii  ├────────┴──► [1] main.py → Polyner.train()
  mask.nii    (metal mask)          │             fits the INR to the scan
  DECBCTSpectrum110KVP.mat          ┘                    │
                                                         ├─► model/model_0.pkl
                                                         └─► output/polyner_RANDO_epoch{N}.nii
                                                             output/loss_log.csv, loss_curve.png
                                                                   │
                              [2] reprojection.py ◄────────────────┘
                              ray-marches the trained network to
                              synthesize dense-view projections,
                              optionally zeroing mu inside metal
                                        │
                                        └─► output/proj_dense_360_metalfree.nii
                                                  │
                              [3] astra_recon.py ◄┘
                              FDK / SIRT / CGLS reconstruction of
                              the synthesized sinogram via ASTRA
                                        │
                                        └─► output/recon_sirt_360_metalfree.nii
```

Stage 1 alone produces a reconstruction (the network is sampled on a voxel grid). Stages 2 and 3 exist to produce a *metal-free* volume: the trained network is reprojected with metal voxels forced to zero attenuation, and that synthetic sinogram is reconstructed with a conventional algorithm.

---

## 2. File reference

```
Polyner/
├── README.md
├── requirements.txt               # non-CUDA Python dependencies
├── .gitignore                      # excludes caches, checkpoints, and generated outputs
├── matlab/
│   └── generate_metal_mask_from_sirt.m # SIRT reconstruction → binary metal mask
├── main.py                         # entry point for training; loads config.json, calls Polyner.train
├── Polyner.py                      # the training loop, forward model, and periodic volume readout
├── dataset.py                      # TrainData (ray/projection sampler) and TestData (readout grid)
├── model.py                        # EDS regularizer (named ASE in this implementation)
├── utils.py                        # cone-beam ray generation, rotation matrices, grid coordinates
├── config.json                     # all geometry, training, encoding, and network parameters
├── reprojection.py                 # stage 2: trained model → dense-view sinogram
├── astra_recon.py                  # stage 3: sinogram → volume via ASTRA
├── debug_repro.py                  # diagnostic for an all-zero reprojection output
├── matnifti.py                     # MATLAB-compatible NIfTI reader/writer and metadata API
├── test_matnifti.py                # executable compatibility/regression checks
├── test.py                         # scratch file, not part of the pipeline (see §9)
├── input/
│   └── RANDO_no_implants_1mm/LE/
│       ├── proj_RANDO_Metal_360degrees.nii     # SimpleITK: (360, 148, 45)
│       ├── mask.nii                            # SimpleITK: (32, 200, 200)
│       ├── fanSensorPosition_fanangle_32f.nii  # 148 horizontal detector angles, deg
│       ├── fanSensorPosition_coneangle_32f.nii # 45 vertical detector angles, deg
│       └── DECBCTSpectrum110KVP.mat            # (7, 2) spectrum, col 0 = LE, col 1 = HE
├── model/                          # local checkpoints (contents gitignored)
└── output/                         # local reconstructions and logs (contents gitignored)
```

### `main.py`
Reads `config.json` with `commentjson` (so the config may contain `//` comments and stray `"comment"` keys) and calls `Polyner.train(img_id=i, config=config)` in a loop that currently runs once. `img_id` only affects output filenames.

### `Polyner.py` — `train(img_id, config)`
The whole training procedure lives in this one function.

**Setup.** Derives the four input paths from `config["file"]["in_dir"]`. The projection filename is hardcoded as `proj_RANDO_Metal_360degrees.nii`, and the detector angle filenames as `fanSensorPosition_{fanangle,coneangle}_32f.nii`. `num_angle` is read from the first axis of the projection volume.

**Metal mask.** `mask.nii` is loaded, symmetrically zero-padded out to 819³ (each axis padded by `SOD - size/2` on the low side and one less on the high side), rotated 90° in the first two axes, and inverted so metal becomes 0 and everything else 1. This inverted mask is the per-point weight for the ASE loss: the energy-smoothness prior is applied outside metal and switched off inside it.

**Spectrum.** `DECBCTSpectrum110KVP.mat` holds a 7×2 array. Bins 1 through 6 of column 0 (the low-energy spectrum) are taken and normalized to sum to 1, giving `e_level = 6`. This directly sets the network's output width. Switching to the high-energy spectrum means changing the column index from 0 to 1; changing the number of bins means editing `e_1, e_n`.

**Network.** A `tcnn.NetworkWithInputEncoding` with 3 inputs and `e_level` outputs, configured entirely from the `encoding` and `network` blocks of the config. Adam optimizer with a `StepLR` schedule.

**Forward model.** For a batch of rays, the network is evaluated at every sample point along every ray, producing attenuation `mu` of shape `(batch, num_sample_ray, 2*SOD, e_level)`. The polychromatic line integral is

```
p̂ = -log( Σ_e  s_e · exp( -Δ · Σ_k mu_e(x_k) ) )
```

where `Δ` is `voxel_size`, `s_e` the normalized spectrum weight, and the inner sum runs over the `2*SOD` samples on the ray. This is Beer's law applied per energy bin, then summed over the spectrum before taking the log, which is what makes the model beam-hardening aware.

**Loss.** L1 between `p̂` and the measured projection, plus the paper's energy-dependent smoothness (EDS) term, named `Attenuation_Smootion_Over_Energies_Loss` in this code.

**Readout.** Every `save_epoch` epochs the network state dict is written to `model/model_{img_id}.pkl` and the network is sampled on a `(2·SOD+1)³` grid in chunks of 100,000 points. The result is cropped to `h × w × d` around the center, transposed to `(2, 0, 1)`, and written as `output/polyner_RANDO_epoch{N}.nii`. The per-epoch loss history is also dumped to `output/loss_log.csv` and plotted (log-y) to `output/loss_curve.png`. Matplotlib is forced to the `Agg` backend for headless cluster use.

### `dataset.py`

**`TrainData`** is indexed by gantry angle, so `len(dataset) == num_angle` and one "epoch" is one pass over all views. Detector positions are read from the two angle files, giving `num_det_u × num_det_v` detector pixels. The measured projection volume arrives as `(num_angle, num_det_col, num_det_row)` and is transposed to `(num_det_row, num_det_col, num_angle)` to match the ray ordering. Rays are generated once in `__init__` at angle 0 and cached as `(num_det, 2*SOD, 3)`.

Each `__getitem__` picks `num_sample_ray` detector pixels uniformly at random without replacement, grabs the matching measured values, and rotates the cached rays to the requested gantry angle. Random sampling (rather than the contiguous blocks used earlier) gives better spatial coverage per step and more i.i.d. gradients.

**`TestData`** is a single-item dataset holding the flattened `(h·w·d, 3)` coordinate grid used for volume readout.

### `utils.py`

- **`cone_beam_ray(proj_pos_u, proj_pos_v, SOD)`** builds the sample-point coordinates for every detector pixel. A reference ray is defined as `2*SOD` evenly spaced points from `y = -1` to `y = +1` along the axis, then rotated toward each detector element by its measured fan angle (about z) and cone angle (about x), pivoting about the source at `(0, -1, 0)`. Returns `(num_det_u, num_det_v, 2*SOD, 3)`. Note that the angle arrays are indexed in reverse (`proj_pos_u[n-i-1]`), which is why `astra_recon.py` flips the sinogram axes back.
- **`build_3d_rotation_matrix(theta_u, theta_v, ox, oy, oz)`** composes `T2 @ Rv @ Ru @ T1`, a rotation about an arbitrary pivot in homogeneous coordinates.
- **`grid_coordinate_3d(h, w, d)`** returns the flattened `[-1, 1]` meshgrid used for readout, with `indexing='ij'`.
- **`rotate_ray_3d(xyz, angle)`** applies the gantry rotation about z, counter-clockwise for positive angles.

### `model.py` — `Attenuation_Smootion_Over_Energies_Loss`
Implements the paper's EDS regularizer (called ASE in older project notes and in the class name). Real tissue attenuation varies smoothly with photon energy, so penalizing adjacent-bin differences steers the network away from degenerate multi-energy solutions. Metal is exempt, since its attenuation genuinely swings hard across the spectrum.

The mask is permuted from `(N, C, h, w, d)` to `(N, C, d, h, w)` to satisfy `grid_sample`'s `(N, C, D, H, W)` convention, then sampled with nearest-neighbor interpolation at every ray point. The loss is

```
lambda · Σ ( Σ_e |mu_{e+1} - mu_e| · mask ) / (batch · num_sample_ray · k)
```

### `reprojection.py`
Loads a trained checkpoint and ray-marches it to produce a dense-view sinogram, optionally treating metal as air.

- **`load_metal_mask(...)`** is the fiddly part. It accepts the mask in either `(h, w, d)` or `(d, w, h)` ordering, optionally undoes the axis-1 flip that the readout applies, permutes to `grid_sample` layout, and computes the mask's extent `[lo, hi]` inside the network's `[-1, 1]` frame from the same crop offsets used at readout time. Without that rescaling the mask would be sampled at the wrong place, since the mask covers only the cropped `h × w × d` sub-volume while the network's domain is the full `2·SOD` cube.
- **`points_in_metal(...)`** rescales query points into the mask's local frame and does a nearest-neighbor lookup.
- **`reproject(config, reproject_config)`** builds angle-0 rays once, then for each of `num_angle_dense` angles rotates them, evaluates the network in chunks, zeros `mu` at masked points, and accumulates `voxel_size · Σ mu` per ray. Only one energy channel is used (`energy_idx`, default the middle bin), so the output is a monochromatic sinogram.
- **Data consistency** (`data_consistency: true`) overwrites every `num_angle_dense / num_angle_sparse`-th view with the real measured projection. This is mutually exclusive with metal masking in practice, since the measured views still contain metal; the code prints a warning rather than refusing.

Configured by the `reproject_config` dict at the bottom of the file, not by `config.json`.

### `astra_recon.py`
Reconstructs the synthesized sinogram with [ASTRA Toolbox](https://astra-toolbox.com/). The geometry derivation is the substance here:

- The `fanSensorPosition_*` files store `atan(pixel_pos / SDD)`, so `SDD · tan(angle)` recovers uniform pixel pitch on a flat panel.
- The u angles run from about -13.1° to +0.3°, meaning the detector is offset rather than centered. This forces `cone_vec` geometry instead of the simpler `cone`.
- The current implementation sets `SAD = SOD · voxel_size` and assumes `SDD = 2 · SAD`. This is a code assumption inherited from the normalized `[-1, +1]` ray span; it is not the physical scanner geometry reported in the project manuscript.
- The sinogram's u and v axes are flipped (`sino[:, ::-1, ::-1]`) to undo the reversed indexing in `cone_beam_ray`, then transposed to ASTRA's `(det_row, angle, det_col)` layout.
- Per-projection vectors place the source at `(0, -SAD, 0)` and the detector center at `(u_center, +ODD, v_center)` at angle 0, rotating both CCW about z to match `rotate_ray_3d`.
- The output volume is transposed back to `(h, w, d)` and flipped on axis 1 so it overlays the Polyner reconstruction.

Command-line interface:

```bash
python3 astra_recon.py --config config.json \
                       --sino ./output/proj_dense_360_metalfree.nii \
                       --out_dir ./output \
                       --out_name recon_sirt_360_metalfree \
                       --algorithm SIRT3D_CUDA \
                       --n_iter 150 \
                       --gpu 0
```

`--algorithm` accepts `FDK_CUDA`, `SIRT3D_CUDA`, or `CGLS3D_CUDA`. FDK ignores `--n_iter` and runs a single pass.

> [!WARNING]
> The VFINAF manuscript reports physical distances of SOD = 410 mm and SDD = 620 mm, so ODD = 210 mm. In consistent centimetres these are 41, 62, and 21 cm. This code expresses `voxel_size` in centimetres, so the checked-in `voxel_size = 0.1` is 1 mm and `astra_recon.py` constructs SAD = 41 cm. It still assumes SDD = 82 cm, not the physical 62 cm. Do not treat its present ASTRA geometry as a faithful physical model until SDD is configured independently.

### `debug_repro.py`
Diagnostic for the case where reprojection produces an all-zero sinogram. It prints per-tensor statistics from the checkpoint's state dict, probes the network at the origin, near the origin, and across the full `[-1, 1]` cube, then runs one angle of ray generation and reports the coordinate ranges plus the fraction of non-zero `mu`. Run it from the repository root with a checkpoint at `model/model_0.pkl`. It hardcodes `cuda:0`.

### `matnifti.py` and `test_matnifti.py`

`matnifti.py` is a NumPy-only, MATLAB-compatible implementation of `niftiwrite`, `niftiread`, and `niftiinfo`. It preserves MATLAB array indexing, supports NIfTI-1 and NIfTI-2, `.nii.gz`, `.hdr/.img`, little/big endian data, scaling metadata, and qform/sform transforms. Use it when an array needs to round-trip between Python and MATLAB without an implicit transpose or flip.

Run the regression suite from the repository root:

```bash
python3 test_matnifti.py
```

`nibabel` is optional; when installed, the suite also cross-checks interoperability with it.

---

## 3. Geometry and data conventions

### Physical CNT CBCT geometry from the VFINAF manuscript

The manuscript reports scanner dimensions in millimetres and attenuation display values in `cm^-1` where applicable.

| Quantity | Reported value |
|---|---|
| Acquisition | 360 cone-beam half-detector projections over 360° |
| Source-to-object distance (SOD/SAD) | 410 mm |
| Source-to-detector distance (SDD) | 620 mm |
| Object-to-detector distance (ODD, derived) | 210 mm |
| X-ray cone angle | 10.4° |
| Flat-panel active area | 147.1 mm × 113.7 mm |
| Native detector pitch | 99 micrometres = 0.099 mm |
| Acquisition binning | 2 × 2; effective pitch approximately 0.198 mm |
| Lateral detector shift | 70 mm |
| FOV at rotation center | 187 mm × 70 mm |
| Source setting | 110 kV, 11 mA, 5 ms per source |
| Filtration | 1.7 mm Al inherent + 0.3 mm Cu external |
| Physical-phantom reconstruction | 0.4 mm isotropic voxels |
| VFINAF virtual reprojection | 1,440 full-detector fan-beam views |

For the paper's digital FORBILD-Defrise simulation, the geometry was SOD = 298 mm, SDD = 567 mm, a 20° cone angle, 360 acquired views, 1,440 virtual views, and 0.5 mm isotropic reconstruction.

### Checked-in sample and normalized code geometry

| Quantity | Value | Source |
|---|---|---|
| Detector type | Flat panel, offset in u (half-fan) | derived from the angle files |
| Detector elements | 148 (u) × 45 (v) | shape of the angle files |
| Fan (u) angle range | -13.12° to +0.32° | `fanSensorPosition_fanangle_32f.nii` |
| Cone (v) angle range | -2.03° to +2.03° | `fanSensorPosition_coneangle_32f.nii` |
| Projection views | 360 over a full turn | first axis of the projection volume |
| `SOD` | 410 | `config.json` |
| `voxel_size` | 0.1 cm = 1.0 mm | `config.json` and project clarification |
| ASTRA SAD | `SOD · voxel_size` = 41.0 | `astra_recon.py` |
| ASTRA SDD | `2 · SAD` = 82.0 (known mismatch with paper) | `astra_recon.py` |
| Spectrum | 110 kVp, 6 of 7 bins, LE column | `DECBCTSpectrum110KVP.mat` |

The code's internal length convention is centimetres: `voxel_size = 0.04` means 0.04 cm = 0.4 mm, while the checked-in `voxel_size = 0.1` means 1.0 mm. ASTRA does not attach unit labels to geometry coordinates: source positions, detector-center positions, per-pixel `u`/`v` vectors, and volume extents must all use the same length scale. The physical manuscript values may therefore be expressed consistently as either 410/620/210 mm or 41/62/21 cm, but they must not be mixed with the current 41/82 assumption.

For NIfTI metadata, convert the internal spacing from centimetres to millimetres: `spacing_mm = 10 * voxel_size`. Thus 0.04 should be written as 0.4 mm and 0.1 as 1.0 mm. `astra_recon.py` currently passes the raw centimetre value to `SimpleITK.SetSpacing`, so its output spacing metadata is ten times too small for software that interprets NIfTI spatial units as millimetres.

`SOD` also sets computational sampling density in this code: every ray gets exactly `2·SOD` = 820 samples and the readout grid is `(2·SOD+1)^3`. Changing it therefore changes both the scanner-like geometry and the cost of a training step.

### Coordinate frame

- Normalized `[-1, 1]` on all three axes, origin at volume center.
- Source at `(0, -1, 0)`; the detector sits toward `+y`.
- Gantry rotates about z, counter-clockwise for positive angles.
- Volume convention is `h → x`, `w → y`, `d → z`.
- `grid_sample` needs `(N, C, D, H, W)` with the grid's last dimension ordered `(x, y, z)`, which is why both `model.py` and `reprojection.py` permute before sampling.

### Array orderings

SimpleITK returns arrays reversed relative to the file's own axis order, which is the source of most of the transposes in this codebase:

| File | As read by SimpleITK | Interpretation |
|---|---|---|
| `proj_*.nii` | `(360, 148, 45)` | `(num_angle, num_det_u, num_det_v)`, transposed to `(v, u, angle)` in `TrainData` |
| `mask.nii` | `(32, 200, 200)` | `(d, w, h)`, depth-major |
| `fanSensorPosition_*` | `(N, 1)` | flattened to `(N,)` |
| reprojection output | `(360, 148, 45)` | `(num_angle, num_det_u, num_det_v)` |

### Sampling arithmetic

With the shipped config: 360 angles at `batch_size = 40` gives 9 optimizer steps per epoch. Each step evaluates `40 × 16 × 820 ≈ 5.2 × 10⁵` network queries. One epoch touches `360 × 16 = 5760` of the `360 × 6660 ≈ 2.4 × 10⁶` available rays, about 0.24%, so the 40,000-epoch default amounts to roughly 96 effective passes over the full projection data.

---

## 4. Configuration reference

`config.json` is parsed with `commentjson`, so comments and extra keys are tolerated.

### `file`

| Key | Default | Meaning |
|---|---|---|
| `in_dir` | `./input/RANDO_no_implants_1mm/LE` | directory holding projections, masks, angle files, spectrum |
| `model_dir` | `./model` | checkpoint destination (must already exist; not auto-created) |
| `out_dir` | `./output` | reconstructions and logs (auto-created) |
| `voxel_size` | `0.1` | ray integration step `Δ` in centimetres; `0.1` = 1.0 mm and `0.04` = 0.4 mm |
| `SOD` | `410` | used as both a scanner-distance-like value and half the ray-sample count; physical manuscript SOD is 410 mm |
| `h`, `w`, `d` | `200, 200, 32` | cropped output volume dimensions |

### `train`

| Key | Default | Meaning |
|---|---|---|
| `gpu` | `0` | CUDA device index |
| `lr` | `1e-3` | initial Adam learning rate |
| `epoch` | `40000` | total epochs; one epoch is one pass over all gantry angles |
| `save_epoch` | `4000` | checkpoint, reconstruct, and re-plot the loss curve at this interval |
| `num_sample_ray` | `16` | detector pixels sampled per angle per step |
| `lr_decay_epoch` | `4000` | `StepLR` step size |
| `lr_decay_coefficient` | `0.5` | `StepLR` gamma |
| `batch_size` | `40` | gantry angles per optimizer step |
| `lambda` | `0.2` | EDS/ASE smoothness-loss weight |

The config carries an experiment note that batch size and `lr_decay_epoch` may have a square-root relationship. Separate project notes tried both 2,000- and 4,000-epoch decay intervals during 40,000-epoch runs. Treat those as empirical starting points, not a scanner-independent rule.

### `encoding` and `network`

Passed straight through to tiny-cuda-nn. The defaults are a 16-level hash grid with 8 features per level, `log2_hashmap_size` 19, base resolution 2, per-level scale 2, and linear interpolation, feeding a `FullyFusedMLP` with 2 hidden layers of 128 neurons, ReLU activations, and a `Squareplus` output. `Squareplus` matters: it keeps predicted attenuation coefficients non-negative without ReLU's dead-gradient problem. Commit history notes that 2 layers of 64 neurons was the best previously tested variant.

The network output width is *not* in the config. It is `e_level`, fixed by the spectrum slice in `Polyner.py`.

### Paper baseline versus this branch

| Setting | Wu et al. paper | Checked-in UNC config |
|---|---|---|
| Geometry | primarily 2D fan beam; one 3D cone-beam mouse-thigh result | 3D UNC cone beam with measured u/v angle files |
| Network | hash encoding, 2 hidden layers x 128 neurons | same default architecture, now with 3D coordinates |
| Rays per optimizer update | 80 random X-rays | `40 angles x 16 detector rays = 640` rays |
| Initial learning rate | `1e-3` | `1e-3` |
| LR decay | x `0.5` every 1,000 epochs | x `0.5` every 4,000 epochs |
| EDS weight | `lambda = 0.2` | `lambda = 0.2` |
| Nominal epochs | 4,000 | 40,000 |

The paper reports that the polychromatic model added about 3.92 dB PSNR over its monochromatic ablation and that `lambda = 0.2` was best on its DeepLesion ablation. Those values motivate the defaults but do not validate them on UNC data.

---

## 5. Requirements and installation

Developed against Python 3.9.6 and PyTorch 2.7. A CUDA GPU is required, since tiny-cuda-nn and the ASTRA CUDA algorithms have no CPU fallback.

| Package | Used by |
|---|---|
| `torch` (CUDA build) | everything |
| `tinycudann` | network definition |
| `numpy` | everything |
| `SimpleITK` | all NIfTI I/O |
| `scipy` | reading the `.mat` spectrum |
| `commentjson` | config parsing in `main.py`, `reprojection.py`, `astra_recon.py` |
| `tqdm` | progress bars |
| `matplotlib` | loss curve plot |
| [`astra-toolbox`](https://astra-toolbox.com/) | FDK/SIRT/CGLS reconstruction and project baselines |
| `scikit-image` | imported in `Polyner.py`, currently unused |

```bash
# 1. PyTorch with CUDA (pick the wheel matching your toolkit)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. tiny-cuda-nn (compiles against your local CUDA toolkit)
pip3 install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# 3. Everything else
pip3 install -r requirements.txt

# 4. ASTRA (the Colab workflow used pip)
pip3 install astra-toolbox

# Or use the official conda channels for a CUDA-enabled environment
conda install -c astra-toolbox -c nvidia astra-toolbox
```

Verify:

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python3 -c "import tinycudann; print('tinycudann ok')"
python3 -c "import astra; print(astra.__version__); astra.test()"
```

Practical minimums: 8 GB VRAM (16 GB+ preferred), 16 GB system RAM (32 GB+ preferred). Reduce `batch_size` and `num_sample_ray` first if you hit OOM during training; reduce `chunk_size` in `reprojection.py` or `Polyner.py` if you hit it during readout.

### Google Colab setup used for the experiments

The project used the following Colab cells. The URLs below are plain shell URLs; the Markdown-link form sometimes copied from notebooks will not work in `pip` or `git` commands.

```python
!pip install torch torchvision torchaudio
!pip install ninja "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
!pip install SimpleITK tqdm numpy scipy scikit-image commentjson astra-toolbox matplotlib

!git clone https://github.com/jperdomounc/Polyner.git
%cd /content/Polyner
!mkdir -p output model
!ls
```

The historical February 2026 experiment cloned with `-b zr-360-feb4`; that branch does not contain the later random-sampling, reprojection, ASTRA, NIfTI compatibility, or documentation work now on `main`. The package commands are intentionally recorded as used, but they are unpinned and may resolve to different versions in a future Colab runtime.

### MATLAB metal-mask generation

Requirements:

- MATLAB R2017b or newer for the core NIfTI workflow. The optional plots use `xline` and `sgtitle`, which require R2018b or newer.
- Image Processing Toolbox for `imfill`, `bwconncomp`, `strel`, `imclose`, `imdilate`, `multithresh`, and visualization helpers.
- Statistics and Machine Learning Toolbox for the `prctile`-based percentile and Otsu-tail modes.

Open `matlab/generate_metal_mask_from_sirt.m`, update `input_nifti`, `output_mask`, the threshold method, and morphology parameters in the **User parameters** block, then run the script in MATLAB. It supports manual, percentile, and Otsu thresholds and shows histogram, overlay, and optional 3D mask visualizations. With `compress_output = true`, MATLAB writes compressed `.nii.gz` output; the current training code expects the exact filename `mask.nii`, so either disable compression or copy/decompress the validated mask into the selected dataset's `LE/` directory. Confirm its dimensions, orientation, and overlay before training or metal-free reprojection.

---

## 6. Running the pipeline

All scripts assume the repository root is the working directory because paths in the config are relative. After cloning, enter the repository once:

```bash
cd /path/to/Polyner
```

The tracked `.gitkeep` files create `model/` and `output/` in a fresh clone; generated contents remain local and are ignored by Git.

**Stage 1: train.**

```bash
python3 main.py
```

Produces `model/model_0.pkl`, `output/polyner_RANDO_epoch{N}.nii` every `save_epoch` epochs, `output/loss_log.csv`, and `output/loss_curve.png`. The progress bar shows the current learning rate and mean epoch loss.

**Stage 2: reproject.** Edit the `reproject_config` dict at the bottom of `reprojection.py` (there is no CLI), then:

```bash
python3 reprojection.py
```

Key fields: `num_angle_dense` (output view count), `model_path`, `out_name`, `metal_mask_path` (set to `None` to keep metal), `energy_idx` (which energy channel to read out), and `data_consistency` with its companion `sparse_proj_path` / `num_angle_sparse`.

**Stage 3: reconstruct.**

```bash
python3 astra_recon.py --sino ./output/proj_dense_360_metalfree.nii \
                       --algorithm SIRT3D_CUDA --n_iter 150
```

---

## 7. Viewing and evaluating results

Output volumes are NIfTI. [ITK-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.SNAP4) and [3D Slicer](https://www.slicer.org/) both open them directly. `astra_recon.py` currently writes the raw centimetre-valued `voxel_size` as its output spacing instead of converting it to millimetres; see the known issue below. The Polyner readout does not set spacing, so `polyner_RANDO_epoch{N}.nii` displays with unit spacing.

Training loss is the only metric currently computed. `output/loss_curve.png` is a log-scale plot of mean epoch loss, rewritten at each save interval, and `output/loss_log.csv` holds the same data per epoch. There is no PSNR/SSIM evaluation in the codebase at present; comparing against a ground-truth volume requires adding it.

Because the loss combines data consistency and ASE, a flat total loss does not by itself distinguish "converged" from "the regularizer is dominating." Sanity-check the reconstructed volume at save points rather than trusting the curve alone.

---

## 8. Adapting to a new dataset

1. Place `proj*.nii`, `mask.nii`, both `fanSensorPosition_*` files, and the spectrum `.mat` under a new directory, and point `in_dir` at it.
2. Update the hardcoded filenames near the top of `Polyner.train` if yours differ. `proj_RANDO_Metal_360degrees.nii` and the `_32f` suffix on the angle files are literal strings in the source, and `reprojection.py` and `astra_recon.py` each repeat the angle filenames independently.
3. Set `h`, `w`, `d` to your reconstruction volume, and confirm `mask.nii` matches: it is read as `(d, w, h)`.
4. Adjust `SOD` and `voxel_size` to your scanner. Remember `SOD` also controls samples per ray and readout grid size, so a large increase is expensive. The current code has no independent `SDD` setting and assumes `SDD = 2 · SAD`; fix or verify that assumption for any physical reconstruction.
5. If your spectrum has a different bin count or you want the high-energy column, edit `e_1`, `e_n`, and the column index in `Polyner.train`, then set a matching `e_level` in `reproject_config`.
6. Output filenames in `Polyner.train` (`polyner_RANDO_epoch{N}.nii`) are also hardcoded.

---

## 9. Known issues and rough edges

These are all present in the tree as of this writing, and worth knowing before you debug something that is already broken.

**An earlier reprojection output was all zeros.** That generated NIfTI is no longer tracked. If the problem recurs, `debug_repro.py` isolates the likely causes: an untrained or badly loaded checkpoint, a network outputting zero, rays falling outside the trained region, or the metal mask zeroing everything. Run it before trusting stage 2 output.

**Three different crop conventions exist.** `Polyner.py` crops with `kx = int(((2·SOD) - h)/2)`, while `test.py` and `reprojection.py`'s mask loader use `kx = int(1 + ((2·SOD) - h)/2)`. A one-voxel offset between the reconstruction and the metal mask follows from this.

**Flip and transpose conventions disagree across files.** `Polyner.py` applies `transpose(2, 0, 1)` with the `np.flip(axis=1)` line commented out. `astra_recon.py` applies `flip(axis=1)`. `reprojection.py`'s `mask_undo_y_flip` defaults to `True`, meaning it assumes the mask was drawn against a flipped volume. If your mask does not line up with your reconstruction, this trio is the place to look.

**Mask padding is off by one relative to the sampling grid.** `Polyner.py` pads the mask to 819³ while the ray sample grid spans `2·SOD = 820` points and the readout grid is 821³. Since `grid_sample` normalizes to the tensor's own extent, this introduces a sub-voxel shift in where the ASE mask is applied.

**`test.py` is not part of the pipeline.** It contains a shape-probing snippet pointing at `./input/RANDO_no_implants_3mm/LE/proj.nii`, which does not exist in this tree, followed by a stale copy of `train()` with the mask disabled (`mask = torch.zeros(...)`) and different hardcoded filenames. Treat it as a scratch file. The crop offset and flip in its copy of `train()` differ from the live `Polyner.py`.

**Unused imports and parameters.** `Polyner.py` imports `erosion` and `square` from `scikit-image` without using them. `TrainData` accepts `voxel_size` and never touches it. `Polyner.py` declares `img_all = []` and never fills it.

**`model_dir` is not created.** `out_path` gets an `os.makedirs`; `model_path` does not, so training crashes at the first save interval if `model/` is missing.

**Reprojection and reconstruction are not configured from `config.json`.** Stage 2's settings live in a dict at the bottom of `reprojection.py`; stage 3's come from CLI arguments. Only geometry is shared with the training config.

**The ASTRA distance ratio does not match the project paper.** The paper establishes millimetres as the physical scanner unit and reports SOD/SDD = 410/620 mm. `astra_recon.py` currently derives SDD as twice SAD, equivalent to 410/820 in the same scale. Add an independent SDD/ODD configuration and verify detector pitch/offset scaling before using ASTRA output quantitatively.

**ASTRA output spacing needs a centimetre-to-millimetre conversion.** `astra_recon.py` passes `voxel_size` directly to `SimpleITK.SetSpacing`. Because this project expresses `voxel_size` in centimetres, the NIfTI spacing should instead be `10 * voxel_size`: 0.04 becomes 0.4 mm and 0.1 becomes 1.0 mm.

**This branch is not the complete VFINAF pipeline.** The project manuscript's key operation is slice-wise dense 2D fan-beam reprojection followed by 2D FBP or SIRT. The current scripts synthesize and reconstruct a 3D cone-beam sinogram instead.

**Inline comments are partly in Chinese**, inherited from the upstream implementation.

### Reproducibility details still needed

The Colab installation commands, SIRT-to-metal-mask MATLAB script, and physical scanner geometry are now recorded. A fully repeatable end-to-end reconstruction still needs the scripts or commands used to create the projection and detector-angle files, any other MATLAB reconstruction steps, pinned CUDA/PyTorch/tiny-cuda-nn/ASTRA versions or the original Colab notebook, a verified mapping between the physical 410/620 mm scanner distances and the network's normalized sampling grid, and identification of the checkpoint/output considered the canonical final result.

---

## 10. License

Available for non-commercial research and education only. Not to be reproduced, exchanged, sold, or used for profit.

## 11. Citation

The project manuscript is *Volumetric Fan-Beam Imaging with Neural Attenuation Field for Cone-Beam Artifact Reduction*. Add its final author list, venue, DOI, and publication year here when those citation details are finalized.

The original Polyner method and reference implementation are by:

```bibtex
@inproceedings{
wu2023unsupervised,
title={Unsupervised Polychromatic Neural Representation for {CT} Metal Artifact Reduction},
author={Qing Wu and Lixuan Chen and Ce Wang and Hongjiang Wei and S Kevin Zhou and Jingyi Yu and Yuyao Zhang},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=xx3QgKyghS}
}
```
