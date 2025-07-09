# Polyner (UNC 3D ms-array adaptation)

This repository is an adaptation the NeurIPS 2023 paper "*Unsupervised Polychromatic Neural Representation for CT Metal Artifact Reduction*" [[OpenReview](https://openreview.net/forum?id=xx3QgKyghS)], [[arXiv](https://arxiv.org/abs/2306.15203)]

![image](gif/fig_method.jpg)
*Fig. 1: Overview of the proposed Polyner model.*

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

## 3. Main Requirements
To run this project, you will need the following packages:
- PyTorch 3.8.13
- tinycudann
- SimpleITK, tqdm, numpy, and other packages.

## 4. Training and Checkpoints

To train our Polyner from scratch, navigate to `./` and run the following command in your terminal:
```shell
python main.py
```
This will train the Polyner model for the metal-corrputed sinogram (`./input/ma_sinogram_0~9.nii`). The well-trained model will be stored in `./model` and its corresponding MAR results will be stored in `./output`.

## 5. Evaluation

To qualitatively evalute the result, navigate to `./` and run the following comman in your terminal:
```shell
python eval.py
```
This will compute PSNR and SSIM values of FBP and our Polyner on the ten samples of the DeepLesion dataset.

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
