#!/usr/bin/env python3
"""
This script provides metal mask thresholding functionality for CT images,
specifically designed to work with multi-source array cone beam CT systems.
It includes both simple thresholding and advanced morphological operations.
"""


import numpy as np
import SimpleITK as sitk
import argparse
import os
from pathlib import Path
from scipy import ndimage
from skimage import morphology, filters, segmentation
import json


class MetalMaskThresholder:
    """
    A class for creating metal masks from CT images using various thresholding techniques.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the MetalMaskThresholder.
        
        Args:
            config_path (str, optional): Path to configuration file
        """
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path):
        """Load configuration parameters."""
        default_config = {
            "metal_threshold_hu": 3000,  # Hounsfield Units for metal detection
            "bone_threshold_hu": 1500,   # Bone threshold to avoid false positives
            "water_threshold_hu": 100,   # Water threshold
            "morphology": {
                "erosion_radius": 1,
                "dilation_radius": 3,
                "closing_radius": 2
            },
            "connected_components": {
                "min_size": 50,  # Minimum size of metal objects (pixels)
                "connectivity": 2  # 2D connectivity (4 or 8)
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
                
        return default_config
    
    def hu_to_linear_attenuation(self, hu_image, mu_water=0.192):
        """
        Convert Hounsfield Units to linear attenuation coefficients.
        
        Args:
            hu_image (np.ndarray): Image in Hounsfield Units
            mu_water (float): Linear attenuation coefficient of water at 70 keV
            
        Returns:
            np.ndarray: Linear attenuation coefficient image
        """
        return (hu_image / 1000.0) * mu_water + mu_water
    
    def simple_threshold_mask(self, image, threshold_hu=3000):
        """
        Create a binary metal mask using simple thresholding.
        
        Args:
            image (np.ndarray): Input CT image (in HU)
            threshold_hu (float): Threshold value in Hounsfield Units
            
        Returns:
            np.ndarray: Binary metal mask
        """
        return (image >= threshold_hu).astype(np.uint8)
    
    def adaptive_threshold_mask(self, image):
        """
        Create a metal mask using adaptive thresholding with Otsu's method.
        
        Args:
            image (np.ndarray): Input CT image
            
        Returns:
            np.ndarray: Binary metal mask
        """
        # Only consider high-intensity regions for Otsu thresholding
        high_intensity_mask = image > self.config["bone_threshold_hu"]
        
        if np.sum(high_intensity_mask) == 0:
            return np.zeros_like(image, dtype=np.uint8)
        
        # Apply Otsu thresholding to high-intensity regions
        threshold = filters.threshold_otsu(image[high_intensity_mask])
        threshold = max(threshold, self.config["metal_threshold_hu"])
        
        return (image >= threshold).astype(np.uint8)
    
    def morphological_processing(self, binary_mask):
        """
        Apply morphological operations to clean up the binary mask.
        
        Args:
            binary_mask (np.ndarray): Input binary mask
            
        Returns:
            np.ndarray: Processed binary mask
        """
        config = self.config["morphology"]
        
        # Erosion to remove noise
        if config["erosion_radius"] > 0:
            kernel = morphology.disk(config["erosion_radius"])
            binary_mask = morphology.binary_erosion(binary_mask, kernel)
        
        # Closing to fill gaps
        if config["closing_radius"] > 0:
            kernel = morphology.disk(config["closing_radius"])
            binary_mask = morphology.binary_closing(binary_mask, kernel)
        
        # Dilation to restore original size and ensure coverage
        if config["dilation_radius"] > 0:
            kernel = morphology.disk(config["dilation_radius"])
            binary_mask = morphology.binary_dilation(binary_mask, kernel)
        
        return binary_mask.astype(np.uint8)
    
    def remove_small_objects(self, binary_mask):
        """
        Remove small connected components that are likely noise.
        
        Args:
            binary_mask (np.ndarray): Input binary mask
            
        Returns:
            np.ndarray: Cleaned binary mask
        """
        config = self.config["connected_components"]
        
        # Remove small objects
        cleaned_mask = morphology.remove_small_objects(
            binary_mask.astype(bool), 
            min_size=config["min_size"],
            connectivity=config["connectivity"]
        )
        
        return cleaned_mask.astype(np.uint8)
    
    def create_metal_mask(self, image, method="adaptive", apply_morphology=True):
        """
        Create a metal mask from CT image.
        
        Args:
            image (np.ndarray): Input CT image
            method (str): Thresholding method ("simple" or "adaptive")
            apply_morphology (bool): Whether to apply morphological processing
            
        Returns:
            np.ndarray: Binary metal mask
        """
        if method == "simple":
            mask = self.simple_threshold_mask(image, self.config["metal_threshold_hu"])
        elif method == "adaptive":
            mask = self.adaptive_threshold_mask(image)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if apply_morphology:
            mask = self.morphological_processing(mask)
            mask = self.remove_small_objects(mask)
        
        return mask
    
    def process_image_file(self, input_path, output_path, method="adaptive"):
        """
        Process a single image file.
        
        Args:
            input_path (str): Path to input image file
            output_path (str): Path to output mask file
            method (str): Thresholding method
        """
        # Read image
        print(f"Reading image: {input_path}")
        image_sitk = sitk.ReadImage(input_path)
        image_array = sitk.GetArrayFromImage(image_sitk)
        
        # Handle 3D images by processing slice by slice
        if len(image_array.shape) == 3:
            mask_array = np.zeros_like(image_array, dtype=np.uint8)
            for i in range(image_array.shape[0]):
                mask_array[i] = self.create_metal_mask(image_array[i], method)
        else:
            mask_array = self.create_metal_mask(image_array, method)
        
        # Create output image
        mask_sitk = sitk.GetImageFromArray(mask_array)
        mask_sitk.CopyInformation(image_sitk)
        
        # Write result
        print(f"Writing mask: {output_path}")
        sitk.WriteImage(mask_sitk, output_path)
        
        # Print statistics
        total_voxels = np.prod(mask_array.shape)
        metal_voxels = np.sum(mask_array)
        print(f"Metal voxels: {metal_voxels}/{total_voxels} ({100*metal_voxels/total_voxels:.2f}%)")
    
    def batch_process(self, input_dir, output_dir, pattern="*.nii", method="adaptive"):
        """
        Process multiple images in a directory.
        
        Args:
            input_dir (str): Input directory path
            output_dir (str): Output directory path
            pattern (str): File pattern to match
            method (str): Thresholding method
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        files = list(input_path.glob(pattern))
        
        if not files:
            print(f"No files found matching pattern {pattern} in {input_dir}")
            return
        
        print(f"Processing {len(files)} files...")
        
        for file_path in files:
            output_file = output_path / f"mask_{file_path.stem}.nii"
            try:
                self.process_image_file(str(file_path), str(output_file), method)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Metal Mask Thresholding for Multi-Source Array Cone Beam CT"
    )
    
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory")
    parser.add_argument("-m", "--method", choices=["simple", "adaptive"], 
                       default="adaptive", help="Thresholding method")
    parser.add_argument("-c", "--config", help="Configuration file path")
    parser.add_argument("-t", "--threshold", type=float, default=3000,
                       help="Metal threshold in HU (for simple method)")
    parser.add_argument("--batch", action="store_true", 
                       help="Batch process directory")
    parser.add_argument("--pattern", default="*.nii", 
                       help="File pattern for batch processing")
    
    args = parser.parse_args()
    
    # Initialize thresholder
    thresholder = MetalMaskThresholder(args.config)
    
    # Override threshold if specified
    if args.threshold != 3000:
        thresholder.config["metal_threshold_hu"] = args.threshold
    
    if args.batch:
        # Batch processing
        output_dir = args.output or f"{args.input}_masks"
        thresholder.batch_process(args.input, output_dir, args.pattern, args.method)
    else:
        # Single file processing
        output_file = args.output or f"mask_{Path(args.input).stem}.nii"
        thresholder.process_image_file(args.input, output_file, args.method)


if __name__ == "__main__":
    main()