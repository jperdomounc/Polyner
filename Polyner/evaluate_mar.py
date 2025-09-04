#!/usr/bin/env python3
"""
CT Metal Artifact Reduction Evaluation Script

This script evaluates metal artifact reduction results by computing PSNR and SSIM
between metal-affected input images and algorithm outputs in non-metal regions.

Author: Generated for Polyner evaluation
Date: 2025-08-21
"""

import os
import re
import numpy as np
import SimpleITK as sitk
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from typing import List, Tuple, Dict, Optional
import glob


class MAREvaluator:
    """Metal Artifact Reduction Evaluator"""
    
    def __init__(self, eval_dir: str = "eval"):
        """
        Initialize the evaluator
        
        Args:
            eval_dir: Directory containing evaluation files
        """
        self.eval_dir = eval_dir
        self.results = []
        
    def find_triplets(self) -> Dict[int, Dict[str, str]]:
        """
        Find all valid triplets of evaluation files
        
        Returns:
            Dictionary mapping case indices to file paths
        """
        if not os.path.exists(self.eval_dir):
            raise FileNotFoundError(f"Evaluation directory '{self.eval_dir}' not found")
            
        # Find all files in eval directory
        all_files = glob.glob(os.path.join(self.eval_dir, "*.nii"))
        
        triplets = {}
        
        # Group files by type and index
        ma_files = {}
        mask_files = {}
        polyner_files = {}
        
        for file_path in all_files:
            filename = os.path.basename(file_path)
            
            # Extract index from filename
            if filename.startswith("1_ma_"):
                match = re.search(r"1_ma_(\d+)", filename)
                if match:
                    idx = int(match.group(1))
                    ma_files[idx] = file_path
                    
            elif filename.startswith("2_mask_"):
                match = re.search(r"2_mask_(\d+)", filename)
                if match:
                    idx = int(match.group(1))
                    mask_files[idx] = file_path
                    
            elif filename.startswith("3_polyner_"):
                match = re.search(r"3_polyner_(\d+)", filename)
                if match:
                    idx = int(match.group(1))
                    polyner_files[idx] = file_path
        
        # Find complete triplets
        all_indices = set(ma_files.keys()) | set(mask_files.keys()) | set(polyner_files.keys())
        
        for idx in sorted(all_indices):
            triplet = {}
            missing = []
            
            if idx in ma_files:
                triplet['ma'] = ma_files[idx]
            else:
                missing.append(f"1_ma_{idx}.nii")
                
            if idx in mask_files:
                triplet['mask'] = mask_files[idx]
            else:
                missing.append(f"2_mask_{idx}.nii")
                
            if idx in polyner_files:
                triplet['polyner'] = polyner_files[idx]
            else:
                missing.append(f"3_polyner_{idx}_*.nii")
            
            if len(triplet) == 3:
                triplets[idx] = triplet
            else:
                print(f"Warning: Incomplete triplet for case {idx}. Missing: {', '.join(missing)}")
                
        return triplets
    
    def load_and_validate_images(self, triplet: Dict[str, str]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Load and validate a triplet of images
        
        Args:
            triplet: Dictionary with 'ma', 'mask', and 'polyner' file paths
            
        Returns:
            Tuple of (ma_image, mask_image, polyner_image) or None if invalid
        """
        try:
            # Load images
            ma_img = sitk.GetArrayFromImage(sitk.ReadImage(triplet['ma']))
            mask_img = sitk.GetArrayFromImage(sitk.ReadImage(triplet['mask']))
            polyner_img = sitk.GetArrayFromImage(sitk.ReadImage(triplet['polyner']))
            
            # Validate dimensions
            if not (ma_img.shape == mask_img.shape == polyner_img.shape):
                print(f"Error: Dimension mismatch in triplet:")
                print(f"  MA: {ma_img.shape}, Mask: {mask_img.shape}, Polyner: {polyner_img.shape}")
                return None
                
            # Ensure mask is binary
            unique_mask_vals = np.unique(mask_img)
            if not np.array_equal(np.sort(unique_mask_vals), [0, 1]) and not np.array_equal(unique_mask_vals, [0]):
                print(f"Warning: Mask contains non-binary values: {unique_mask_vals}")
                # Binarize mask
                mask_img = (mask_img > 0).astype(np.float32)
            
            return ma_img.astype(np.float32), mask_img.astype(np.float32), polyner_img.astype(np.float32)
            
        except Exception as e:
            print(f"Error loading triplet: {e}")
            return None
    
    def compute_metrics(self, ma_img: np.ndarray, mask_img: np.ndarray, polyner_img: np.ndarray) -> Tuple[float, float]:
        """
        Compute PSNR and SSIM between MA input and Polyner output in non-metal regions
        
        Args:
            ma_img: Metal-affected input image
            mask_img: Binary metal mask (1=metal, 0=non-metal)
            polyner_img: Algorithm output image
            
        Returns:
            Tuple of (PSNR, SSIM)
        """
        # Create non-metal mask (where mask == 0)
        non_metal_mask = (mask_img == 0)
        
        if np.sum(non_metal_mask) == 0:
            print("Warning: No non-metal regions found")
            return 0.0, 0.0
        
        # Apply mask to images
        ma_masked = ma_img * non_metal_mask
        polyner_masked = polyner_img * non_metal_mask
        
        # Calculate PSNR
        # Use data range based on the actual image values
        data_range = max(np.max(ma_img) - np.min(ma_img), np.max(polyner_img) - np.min(polyner_img))
        if data_range == 0:
            psnr = float('inf')
        else:
            psnr = peak_signal_noise_ratio(ma_masked, polyner_masked, data_range=data_range)
        
        # Calculate SSIM
        # For 3D images, calculate SSIM slice by slice and average
        if len(ma_img.shape) == 3:
            ssim_values = []
            for i in range(ma_img.shape[0]):
                slice_mask = non_metal_mask[i]
                if np.sum(slice_mask) > 0:  # Only compute if there are non-metal pixels
                    ssim_slice = structural_similarity(
                        ma_masked[i], polyner_masked[i], 
                        data_range=data_range,
                        mask=slice_mask
                    )
                    ssim_values.append(ssim_slice)
            ssim = np.mean(ssim_values) if ssim_values else 0.0
        else:
            ssim = structural_similarity(
                ma_masked, polyner_masked, 
                data_range=data_range,
                mask=non_metal_mask
            )
        
        return psnr, ssim
    
    def evaluate_case(self, case_idx: int, triplet: Dict[str, str]) -> Optional[Tuple[float, float]]:
        """
        Evaluate a single case
        
        Args:
            case_idx: Case index
            triplet: Dictionary with file paths
            
        Returns:
            Tuple of (PSNR, SSIM) or None if evaluation failed
        """
        print(f"Evaluating Case {case_idx}...")
        
        # Load and validate images
        images = self.load_and_validate_images(triplet)
        if images is None:
            return None
            
        ma_img, mask_img, polyner_img = images
        
        # Compute metrics
        psnr, ssim = self.compute_metrics(ma_img, mask_img, polyner_img)
        
        print(f"Case {case_idx}: PSNR={psnr:.2f}, SSIM={ssim:.5f}")
        
        return psnr, ssim
    
    def run_evaluation(self):
        """Run complete evaluation"""
        print("Starting Metal Artifact Reduction Evaluation...")
        print(f"Looking for files in: {self.eval_dir}")
        
        # Find all triplets
        triplets = self.find_triplets()
        
        if not triplets:
            print("No complete triplets found. Please check your file naming and directory.")
            return
            
        print(f"Found {len(triplets)} complete triplet(s)")
        
        # Evaluate each case
        results = []
        for case_idx in sorted(triplets.keys()):
            result = self.evaluate_case(case_idx, triplets[case_idx])
            if result is not None:
                results.append((case_idx, result[0], result[1]))
        
        if not results:
            print("No cases could be evaluated successfully.")
            return
            
        # Calculate summary statistics
        psnr_values = [r[1] for r in results]
        ssim_values = [r[2] for r in results]
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        for case_idx, psnr, ssim in results:
            print(f"Case {case_idx}: PSNR={psnr:.2f}, SSIM={ssim:.5f}")
        
        if len(results) > 1:
            print(f"\nMean ± Std:")
            print(f"PSNR: {np.mean(psnr_values):.2f} ± {np.std(psnr_values):.2f}")
            print(f"SSIM: {np.mean(ssim_values):.5f} ± {np.std(ssim_values):.5f}")
        
        # Save results to file
        self.save_results(results)
        
        return results
    
    def save_results(self, results: List[Tuple[int, float, float]]):
        """Save results to file"""
        results_file = os.path.join(self.eval_dir, "metrics.txt")
        
        with open(results_file, 'w') as f:
            f.write("Metal Artifact Reduction Evaluation Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Evaluation Directory: {self.eval_dir}\n")
            f.write(f"Number of Cases: {len(results)}\n\n")
            
            f.write("Individual Results:\n")
            for case_idx, psnr, ssim in results:
                f.write(f"Case {case_idx}: PSNR={psnr:.2f}, SSIM={ssim:.5f}\n")
            
            if len(results) > 1:
                psnr_values = [r[1] for r in results]
                ssim_values = [r[2] for r in results]
                
                f.write(f"\nSummary Statistics:\n")
                f.write(f"PSNR: {np.mean(psnr_values):.2f} ± {np.std(psnr_values):.2f}\n")
                f.write(f"SSIM: {np.mean(ssim_values):.5f} ± {np.std(ssim_values):.5f}\n")
        
        print(f"\nResults saved to: {results_file}")


def main():
    """Main function"""
    # You can specify a different eval directory here if needed
    eval_dir = "eval"
    
    evaluator = MAREvaluator(eval_dir)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()