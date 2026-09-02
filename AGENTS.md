# Agent instructions for Polyner

These instructions apply to the entire repository. Read `README.md` for the full method, pipeline, and known-issue history before changing reconstruction behavior.

## What this repository is

Polyner is experimental research code for metal-artifact reduction in UNC multisource-array cone-beam CT. It fits a case-specific tiny-cuda-nn model, optionally reprojects a metal-suppressed sinogram, and reconstructs that sinogram with ASTRA.

- This is not clinically validated software. Never present an output as suitable for diagnosis, treatment, or quantitative interpretation.
- A lower training loss and a successfully written NIfTI file do not establish image correctness.
- Geometry, orientation, units, mask alignment, and image quality require human inspection for every dataset.
- This branch is not the complete VFINAF pipeline described in the project manuscript; see the README.

## Compute environment: do not run the pipeline on a CPU-only laptop

Assume the current computer is not capable of running the reconstruction pipeline unless working CUDA access has been explicitly confirmed.

- Do not run `main.py`, `reprojection.py`, `debug_repro.py`, or `astra_recon.py` locally without a compatible NVIDIA GPU, CUDA-enabled PyTorch, and the required CUDA packages.
- Training and reprojection depend on `tiny-cuda-nn`; the exposed ASTRA algorithms are CUDA implementations. There is no supported CPU pipeline. `Polyner.py` also has a malformed CPU fallback that can construct `cuda:cpu`.
- `requirements.txt` intentionally omits the CUDA PyTorch build, `tiny-cuda-nn`, and ASTRA; follow the GPU-environment notes in the README instead of treating that file as a complete environment specification.
- Use an allocated GPU compute node on UNC Longleaf (or a comparable GPU cluster) over SSH. Do not run heavy work on a cluster login node.
- If no suitable remote GPU is available, use a Google Colab runtime with a GPU. Confirm the runtime is actually using a GPU before installing/building `tiny-cuda-nn` or starting work.
- Do not install or rebuild the CUDA stack on the local computer merely to validate a small code/documentation change.
- Do not start a long or costly training/reconstruction job unless the user asked for it. Record the host/GPU, git commit, environment, config, inputs, checkpoint, and output paths for any real experiment.

Lightweight inspection, editing, syntax checks, and the standalone NIfTI regression can be done locally.

## NIfTI handling is high risk

Treat `matnifti.py` as an experimental compatibility implementation, not an authoritative medical-imaging writer. Its regression tests show compatibility with the included fixtures and several NIfTI variants; they do not prove that a newly produced scan has correct physical geometry or clinical meaning.

- Prefer a person experienced with CT, NIfTI, and MATLAB to create or approve authoritative imaging files.
- The live reconstruction scripts write through SimpleITK, not `matnifti.py`. `Polyner.py` and `reprojection.py` create output images without carrying through meaningful spacing/origin/direction metadata, and `astra_recon.py` has a documented spacing-scale error. Treat these outputs as provisional voxel arrays until validated in physical space.
- Never overwrite an original projection, reconstruction, detector-angle file, spectrum, or mask. Write a new candidate under an explicitly named scratch/output path and preserve provenance.
- Preserve and verify dimensions, datatype, endianness, scaling, pixel spacing, spatial units, qform/sform affine, handedness, description, and compression behavior.
- `matnifti.niftiread` follows MATLAB-like raw-value behavior and does not apply header scaling unless requested. Check scaling explicitly when intensity values matter.
- MATLAB/`matnifti` use column-major data without SimpleITK's array-axis reversal. `SimpleITK.GetArrayFromImage` exposes `(z, y, x)` for a 3-D image. Never replace a required transpose/flip with `reshape`.
- For the checked-in data, the projection is `(45, 148, 360)` in MATLAB/`matnifti` order and `(360, 148, 45)` through SimpleITK. The mask is `(200, 200, 32)` in MATLAB/`matnifti` order and `(32, 200, 200)` through SimpleITK.
- Cross-check important files in MATLAB with `niftiinfo`/`niftiread`, and inspect them in an independent viewer such as ITK-SNAP or 3D Slicer. A byte-identical round trip is useful but is not a substitute for checking the image in physical space.
- Any edit to `matnifti.py` must keep MATLAB-compatible column-major/no-axis-reversal behavior and pass `python3 test_matnifti.py`.

## Metal masks require expert generation and review

`matlab/generate_metal_mask_from_sirt.m` explicitly began as Claude-generated code. Treat it as an unvalidated draft and a possible starting point only. Its default paths are stale, and its percentile, threshold, and morphology settings are not scanner- or dataset-independent.

- Do not autonomously generate, tune, replace, or declare a metal mask correct.
- Prefer a skilled CT/NIfTI/MATLAB user working from the correct SIRT or CT reconstruction, with knowledge of the image's intensity units and scanner geometry.
- The default top-0.5% rule will select voxels even when a volume has no true metal. The example threshold of `3000` is meaningful only for HU-like data, not generic attenuation coefficients.
- Make a new candidate file rather than overwriting `mask.nii`. Before it is used, a human should review overlays slice-by-slice in all three planes and verify dimensions, affine/header, spacing/units, crop, flips, threshold, connected components, dilation, and the fraction of voxels labeled as metal.
- In this code, `mask.nii == 1` denotes metal. Keep candidate masks strictly binary with values `0` and `1`, not `0` and `255` or soft probabilities. Training inverts the mask so the energy-smoothness term applies outside metal; reprojection sets attenuation to zero at masked samples. A wrong or misaligned mask can silently corrupt both stages.
- Require explicit human approval before replacing the tracked mask or launching costly training with a new mask.

## Known correctness hazards

Do not silently “clean up” these conventions. Resolve them with tests and, for geometry changes, domain review.

- `Polyner.py`, `reprojection.py`, and `astra_recon.py` disagree in places about crop offsets, axis flips, and transposes. There is a documented one-voxel crop offset, and mask padding is off by one relative to ray/readout grids.
- `reprojection.py` defaults `mask_undo_y_flip=True`; confirm this against the volume on which the mask was drawn.
- `astra_recon.py` assumes `SDD = 2 * SAD`, while project documentation gives physical SOD/SDD values of 410/620 mm. It also writes a centimetre-valued `voxel_size` directly as NIfTI spacing even though NIfTI viewers generally expect millimetres here.
- Metal masking and measured-view data consistency conflict: measured views still contain metal.
- Training is polychromatic, while current reprojection selects one energy channel. Confirm `e_level` and `energy_idx` match the checkpoint.
- Several input and output names are hardcoded. Check every resolved path and array shape rather than trusting a filename.
- Do not infer acquisition content or provenance from names alone: the checked-in directory says `no_implants`, while the projection filename contains `Metal`.
- `test.py` is stale scratch code, has top-level side effects and missing input paths, and disables the real mask. It is not a test or pipeline entry point.

## Repository workflow

- Run scripts from the repository root; relative paths assume that working directory.
- The live stages are `main.py` -> `reprojection.py` -> `astra_recon.py`. `Polyner.py` contains training, `dataset.py`/`utils.py` contain ray and array conventions, and `model.py` contains the EDS/ASE regularizer.
- Read the “Known issues and rough edges” section of `README.md` before changing geometry, masks, NIfTI output, or physical units.
- Keep changes narrow. Do not change geometry, orientation, energy bins, mask semantics, or physical units as incidental refactors.
- Inspect `git status` before and after work. Preserve existing user changes and do not commit generated checkpoints or outputs.
- `model/` and `output/` contents are ignored except for `.gitkeep`; `input/` is not generally ignored. Never stage a new scan or derived patient data by default.
- Do not treat a pre-existing ignored checkpoint or output as canonical. An all-zero dense reprojection has occurred in this working tree before; verify provenance and statistics from scratch.
- Current tracked inputs are described as RANDO phantom data, but never assume future data is deidentified. NIfTI headers, filenames, screenshots, masks, logs, and case-specific checkpoints can contain or imply sensitive information. Do not upload them to GitHub, public Colab/Drive, chat, issue trackers, or third-party services without explicit authorization and institutionally compliant handling.

## Validation

There is no configured pytest, linter, or formatter suite. Do not run bare `pytest`, because it may collect the stale `test.py`.

Safe local regression from the repository root:

```bash
python3 test_matnifti.py
```

This writes to a temporary directory. Optional nibabel and SimpleITK checks are skipped if those packages are absent. Report skipped checks; do not describe a partial run as full cross-library validation.

For GPU changes, begin with the smallest representative smoke run on an authorized GPU environment before a full experiment. Validate more than process exit status:

- inputs and outputs have the expected shapes and nonzero finite ranges;
- loss and reconstructed values remain finite;
- mask overlays align in all planes;
- spacing, units, qform/sform, and orientation survive writing;
- reprojection is not all zeros; and
- results receive visual review by someone qualified to assess CT reconstructions.

Sampling is stochastic and no seed is currently set. Record that limitation and do not claim exact reproducibility unless seed handling and the full CUDA environment have also been controlled.
