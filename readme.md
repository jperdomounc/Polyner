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
Polyner
│  config.json					# configuration script (original)
│  config_unc.json				# UNC-specific configuration
│  dataset.py					# dataloader
│  eval.py			   		# quantitative evaluation
│  main.py					# running script for training (original)
│  main_unc.py					# UNC-specific training script
│  model.py					# EAS loss
│  readme.md					# readme file
│  Polyner.py					# training function
│  utils.py					# tools
│  prepare_unc_data.py				# UNC data preparation script
│  convert_unc_data.py				# UNC data conversion utilities
│  convert_nrrd_to_nii.py			# NRRD to NIfTI conversion
│  metal_mask_threshold.py			# Metal mask generation
│  requirements.txt				# Python dependencies
│  package.json					# Node.js dependencies
│  notes.md					# development notes
│  notes.txt					# additional notes
│  
├─data_simulation				# data simulation
│  │  config_dl.yaml				# acquisition parameters
│  │  dl_data.m					# running script for DeepLesion dataset
│  │  
│  ├─+helper					# functions for data simulation
│  │      get_mar_params.m
│  │      interpolate_projection.m
│  │      pkev2kvp.m
│  │      simulate_metal_artifact.m
│  │      @YAML/					# YAML parsing utilities
│  │              
│  ├─metal					# prior data for simulation
│  │      GE14Spectrum120KVP.mat
│  │      MiuofAl.mat, MiuofAu.mat, etc.	# material attenuation data
│  │      SampleMasks.mat
│  │      
│  └─slice
│          gt_0.nii to gt_199.nii		# raw data (200 slices)
│      
├─input						# original DeepLesion dataset
│      fanSensorPos.nii				# geometry angle
│      GE14Spectrum120KVP.mat			# energy spectrum
│      gt_0.nii to gt_9.nii			# ground truth images
│      mask_0.nii to mask_9.nii			# metal masks
│      ma_0.nii to ma_9.nii			# FBP reconstructions
│      ma_sinogram_0.nii to ma_sinogram_9.nii	# metal-corrupted measurements
│      
├─input_unc					# UNC-specific input data
│      fanSensorPos.nii				# UNC linear detector geometry
│      GE14Spectrum120KVP.mat			# energy spectrum
│      gt_0.nii to gt_2.nii			# UNC ground truth images
│      mask_0.nii to mask_2.nii			# UNC metal masks
│      ma_0.nii to ma_2.nii			# UNC FBP reconstructions
│      ma_sinogram_0.nii to ma_sinogram_2.nii	# UNC metal-corrupted measurements
│      
├─UNCtestdata					# UNC RANDO phantom data
│  │  config.txt				# acquisition parameters
│  │  Proj_RANDO_Metal_DEMSCBCT_src5_110kvp_744_229.bin	# raw projections
│  │  Rec_RANDO_Metal_DEMSCBCT_src5_110kvp_480_480_120_75keV_HU.bin	# HU reconstruction
│  │  Rec_RANDO_Metal_DEMSCBCT_src5_110kvp_480_480_120_75keV_mu.bin	# μ reconstruction
│  │  Segmentation-Segment_1-label.nrrd		# metal segmentation
│  │  spectrum_UNC.mat				# UNC X-ray spectrum
│  │  slice42_216_216.bin, rec42_216_216.bin, sino42_400_360.bin	# test slices
│  │  Intro1.jpg, Intro2.jpg			# documentation images
│  │  
│  ├─converted					# processed UNC data
│  │      RANDO_Metal_HU_480x480x120.nii	# 3D HU volume
│  │      RANDO_Metal_mu_480x480x120.nii	# 3D μ volume
│  │      metal_mask_RANDO*.nii			# various metal masks
│  │      Segmentation-Segment_1-label.nii	# converted segmentation
│  │      slice42_216x216.nii, rec42_216x216.nii	# test slice data
│  │      
│  └─DualEnergy Result				# dual energy results
│      └─DEMSCBCT
│          └─RANDO_MAR_Ca_110kVp_noconstrain
│              ├─VMI HU				# virtual monoenergetic images (HU)
│              └─VMI mu				# virtual monoenergetic images (μ)
│      
├─model						# trained models (original)
│      model_x.pkl				# pre-trained Polyner
│      
├─model_unc					# UNC-specific trained models
│      
├─output					# original results
│      polyner_0.nii to polyner_9.nii		# Polyner reconstructions
│      
├─output_unc					# UNC-specific results
│      
└─gif						# visualization assets
        fig1.gif, fig2.gif			# result animations
        fig_method.jpg				# method overview
```

## 3. Dependencies and Requirements

### Python Dependencies
The following Python packages are required to run the UNC MAR package:

**Core Dependencies:**
- Python 3.8+
- PyTorch (with CUDA support recommended)
- torchvision
- torchaudio
- tinycudann (tiny-cuda-nn) - Neural network acceleration
- numpy - Numerical computing
- SimpleITK - Medical image processing
- scipy - Scientific computing library

**Additional Dependencies:**
- tqdm - Progress bars
- commentjson - JSON parsing with comments
- scikit-image (skimage) - Image processing metrics (SSIM, PSNR)
- pathlib - Path handling (Python standard library)

### MATLAB Dependencies
For data simulation and preprocessing:
- MATLAB with Image Processing Toolbox
- YAML parser (included in `data_simulation/+helper/@YAML/`)

### System Requirements
- CUDA-compatible GPU (recommended for training)
- Minimum 8GB RAM
- Storage space for datasets and models

### Installation
```bash
pip install torch torchvision torchaudio
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install simpleitk tqdm numpy commentjson scikit-image scipy
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
- **Data Format**: Binary reconstruction files (.bin) converted to NIfTI format
- **Energy Spectrum**: UNC X-ray spectrum data (`spectrum_UNC.mat`)
- **Geometry Configuration**: Linear detector geometry in `fanSensorPos.nii`

### Training for UNC Data
```bash
# Prepare UNC-specific data
python prepare_unc_data.py

# Train with UNC configuration
python main_unc.py

# Evaluate UNC results
python eval.py --config config_unc.json
```

The UNC-specific configuration file (`config_unc.json`) contains parameters optimized for the multisource array geometry and clinical protocols.

## 5. Training and Checkpoints (Original Method)

To train the original Polyner model, navigate to `./` and run:
```shell
python main.py
```
This trains the model on DeepLesion simulation data (`./input/ma_sinogram_0~9.nii`). Models are stored in `./model` and results in `./output`.

## 6. Evaluation

To qualitatively evaluate the results, navigate to `./` and run:
```shell
python eval.py
```
This computes PSNR and SSIM values of FBP and Polyner on the DeepLesion dataset samples.

For the ten sinograms (`./input/ma_sinogram_0~9.nii`), the quantitative results are shown in:

|Method         | PSNR  | SSIM |
|:------------------: |:--------------: | :------------: |
|FBP   | 29.13±3.27 | 0.7201±0.1109 |
|Polyner   | 37.33±0.93 | 0.9774±0.0031 |

## 6. Data Simulation
To simulate the metal-corrupted measurements, navigate to `./data_simulation` and run the MATLAB script `dl_data.m`. These code for data simulation are based on the ADN repository: https://github.com/liaohaofu/adn/tree/master


## 7. Others

NIFTI files (`.nii`) can be viewed by using the ITK-SNAP software, which is available for free download at: http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.SNAP4


## 8. UNC Cone Beam CT Adaptation

This repository has been adapted for UNC's 3D multisource cone beam CT system. Key modifications include:

### UNC-Specific Configuration
- **Geometry**: Linear detector (not arc) with UNC specifications:
  - Source-to-object distance (SOD): 410mm
  - Source-to-detector distance (SDD): 620mm  
  - Detector dimensions: 148.8mm × 148.8mm
  - Detector pixel size: 0.2mm
  - Detector offset: 70.5mm

### Data Preparation
- `prepare_unc_data.py`: Converts UNC RANDO phantom data to Polyner format
- `config_unc.json`: UNC-specific configuration parameters
- `input_unc/`: Directory containing UNC test data

### MATLAB Simulation Updates
- Modified `simulate_metal_artifact.m` to use linear detector geometry
- Updated all `fanbeam`/`ifanbeam` calls from arc to line geometry
- Ensures proper forward/backward projection for UNC system

### Usage for UNC Data
```bash
# Prepare UNC data
python prepare_unc_data.py

# Train with UNC configuration  
python main.py --config config_unc.json

# Evaluate UNC results
python eval.py --config config_unc.json
```

## 9. License

This code is available for non-commercial research and education purposes only. It is not allowed to be reproduced, exchanged, sold, or used for profit.

## 10. Citation

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
