import SimpleITK as sitk
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import os
from pathlib import Path

class PolynerDebugger:
    def __init__(self, config_path="./Polyner/config.json"):
        """Initialize debugger with config"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Fix path - remove extra ./ if present
        self.in_path = self.config['file']['in_dir'].replace('./', '')
        if not self.in_path.startswith('./'):
            self.in_path = './Polyner/' + self.in_path
        self.SOD = self.config['file']['SOD']
        self.h = self.config['file']['h']
        self.w = self.config['file']['w']
        
        print("=== POLYNER DEBUGGING SUITE ===")
        print(f"Config loaded: SOD={self.SOD}, image size=({self.h},{self.w})")
    
    def check_data_consistency(self, img_id=0):
        """Check dimensional consistency between all data files"""
        print(f"\n=== DATA CONSISTENCY CHECK (img_id={img_id}) ===")
        
        # File paths
        mask_path = f'./{self.in_path}/mask_{img_id}.nii'
        ma_path = f'./{self.in_path}/ma_{img_id}.nii'
        gt_path = f'./{self.in_path}/gt_{img_id}.nii'
        sino_path = f'./{self.in_path}/ma_sinogram_{img_id}.nii'
        sensor_path = f'./{self.in_path}/fanSensorPos.nii'
        spectrum_path = f'./{self.in_path}/GE14Spectrum120KVP.mat'
        
        files_info = {}
        
        # Check image files
        for name, path in [('mask', mask_path), ('ma_image', ma_path), ('gt_image', gt_path)]:
            if os.path.exists(path):
                data = sitk.GetArrayFromImage(sitk.ReadImage(path))
                files_info[name] = {
                    'shape': data.shape,
                    'dtype': data.dtype,
                    'range': (data.min(), data.max()),
                    'mean': data.mean()
                }
                print(f"{name:12}: shape={data.shape}, range=[{data.min():.3f}, {data.max():.3f}]")
            else:
                print(f"{name:12}: FILE NOT FOUND - {path}")
        
        # Check sinogram
        if os.path.exists(sino_path):
            sino_data = sitk.GetArrayFromImage(sitk.ReadImage(sino_path))
            files_info['sinogram'] = {
                'shape': sino_data.shape,
                'dtype': sino_data.dtype,
                'range': (sino_data.min(), sino_data.max())
            }
            print(f"sinogram    : shape={sino_data.shape}, range=[{sino_data.min():.3f}, {sino_data.max():.3f}]")
        
        # Check sensor positions
        if os.path.exists(sensor_path):
            sensor_data = sitk.GetArrayFromImage(sitk.ReadImage(sensor_path))
            files_info['sensor'] = {
                'shape': sensor_data.shape,
                'dtype': sensor_data.dtype,
                'range': (sensor_data.min(), sensor_data.max())
            }
            print(f"sensor_pos  : shape={sensor_data.shape}, range=[{sensor_data.min():.3f}, {sensor_data.max():.3f}]")
        
        # Check spectrum
        if os.path.exists(spectrum_path):
            spectrum_data = scio.loadmat(spectrum_path)
            spectrum_key = 'GE14Spectrum120KVP'
            if spectrum_key in spectrum_data:
                spec = spectrum_data[spectrum_key]
                files_info['spectrum'] = {
                    'shape': spec.shape,
                    'energy_range': (spec[0,0], spec[-1,0]),
                    'max_intensity': spec[:,1].max()
                }
                print(f"spectrum    : shape={spec.shape}, energy=[{spec[0,0]:.0f}, {spec[-1,0]:.0f}] keV")
            else:
                print(f"spectrum    : KEY '{spectrum_key}' NOT FOUND in {list(spectrum_data.keys())}")
        
        return files_info
    
    def validate_mask(self, img_id=0, visualize=True):
        """Validate mask properties - Line 50 debugging"""
        print(f"\n=== MASK VALIDATION (img_id={img_id}) ===")
        
        mask_path = f'./{self.in_path}/mask_{img_id}.nii'
        if not os.path.exists(mask_path):
            print(f"ERROR: Mask file not found: {mask_path}")
            return None
        
        # Load mask as Polyner does
        mask_original = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        print(f"Original mask shape: {mask_original.shape}")
        print(f"Original mask range: [{mask_original.min():.6f}, {mask_original.max():.6f}]")
        print(f"Original mask unique values: {np.unique(mask_original)}")
        
        # Apply Polyner's preprocessing (Line 51-52)
        mask_padded = np.rot90(np.pad(mask_original, 
                                    ((int(self.SOD - (mask_original.shape[0] / 2)), 
                                      int(self.SOD - (mask_original.shape[0] / 2))-1),
                                     (int(self.SOD - (mask_original.shape[1] / 2)), 
                                      int(self.SOD - (mask_original.shape[1] / 2))-1))))
        
        print(f"After padding/rotation: {mask_padded.shape}")
        print(f"Padded mask range: [{mask_padded.min():.6f}, {mask_padded.max():.6f}]")
        print(f"Padded mask unique values: {np.unique(mask_padded)}")
        
        # Check if mask has proper binary values
        metal_pixels = np.sum(mask_padded == 1)
        background_pixels = np.sum(mask_padded == 0)
        other_pixels = mask_padded.size - metal_pixels - background_pixels
        
        print(f"Metal pixels (value=1): {metal_pixels} ({100*metal_pixels/mask_padded.size:.2f}%)")
        print(f"Background pixels (value=0): {background_pixels} ({100*background_pixels/mask_padded.size:.2f}%)")
        print(f"Other values: {other_pixels} pixels")
        
        if other_pixels > 0:
            print("⚠️  WARNING: Mask contains non-binary values!")
        
        if metal_pixels == 0:
            print("❌ ERROR: No metal pixels found (mask should have some 1s)")
        else:
            print("✓ Mask contains metal regions")
        
        if visualize:
            self.plot_mask_analysis(mask_original, mask_padded, img_id)
        
        return {
            'original_shape': mask_original.shape,
            'padded_shape': mask_padded.shape,
            'metal_pixels': metal_pixels,
            'background_pixels': background_pixels,
            'other_pixels': other_pixels,
            'is_binary': other_pixels == 0
        }
    
    def plot_mask_analysis(self, mask_original, mask_padded, img_id):
        """Plot mask before and after processing"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original mask
        im1 = axes[0].imshow(mask_original, cmap='gray', origin='lower')
        axes[0].set_title(f'Original Mask (img_id={img_id})')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[0])
        
        # Padded/rotated mask
        im2 = axes[1].imshow(mask_padded, cmap='gray', origin='lower')
        axes[1].set_title('After Padding/Rotation')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        plt.colorbar(im2, ax=axes[1])
        
        # Difference visualization
        center_y, center_x = mask_padded.shape[0]//2, mask_padded.shape[1]//2
        roi_size = min(mask_original.shape[0], 100)
        roi = mask_padded[center_y-roi_size//2:center_y+roi_size//2,
                         center_x-roi_size//2:center_x+roi_size//2]
        
        im3 = axes[2].imshow(roi, cmap='hot', origin='lower')
        axes[2].set_title(f'ROI Center ({roi_size}x{roi_size})')
        axes[2].set_xlabel('X')
        axes[2].set_ylabel('Y')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(f'mask_analysis_img{img_id}.png', dpi=150, bbox_inches='tight')
        print(f"✓ Mask analysis saved as 'mask_analysis_img{img_id}.png'")
        plt.show()
    
    def verify_spectrum(self, plot=True):
        """Verify and plot energy spectrum - Line 58 debugging"""
        print(f"\n=== SPECTRUM VERIFICATION ===")
        
        spectrum_path = f'./{self.in_path}/GE14Spectrum120KVP.mat'
        if not os.path.exists(spectrum_path):
            print(f"ERROR: Spectrum file not found: {spectrum_path}")
            return None
        
        try:
            spectrum_data = scio.loadmat(spectrum_path)
            spectrum_key = 'GE14Spectrum120KVP'
            
            if spectrum_key not in spectrum_data:
                print(f"ERROR: Key '{spectrum_key}' not found in spectrum file")
                print(f"Available keys: {[k for k in spectrum_data.keys() if not k.startswith('__')]}")
                return None
            
            spectrum = spectrum_data[spectrum_key]
            print(f"Spectrum shape: {spectrum.shape}")
            print(f"Energy range: {spectrum[0,0]:.1f} - {spectrum[-1,0]:.1f} keV")
            print(f"Number of energy bins: {spectrum.shape[0]}")
            print(f"Max intensity: {spectrum[:,1].max():.6f}")
            print(f"Total intensity (area under curve): {np.trapz(spectrum[:,1], spectrum[:,0]):.6f}")
            
            # Check for common issues
            if spectrum.shape[1] != 2:
                print("⚠️  WARNING: Spectrum should have 2 columns (energy, intensity)")
            
            if np.any(spectrum[:,0] <= 0):
                print("⚠️  WARNING: Spectrum contains non-positive energies")
            
            if np.any(spectrum[:,1] < 0):
                print("⚠️  WARNING: Spectrum contains negative intensities")
            
            if spectrum[:,1].max() == 0:
                print("❌ ERROR: Spectrum has zero intensity everywhere")
            
            # Check energy spacing
            energy_diffs = np.diff(spectrum[:,0])
            if not np.allclose(energy_diffs, energy_diffs[0], rtol=1e-10):
                print(f"ℹ️  Energy spacing is non-uniform: {energy_diffs[:5]}...")
            else:
                print(f"✓ Uniform energy spacing: {energy_diffs[0]:.1f} keV")
            
            if plot:
                self.plot_spectrum(spectrum)
            
            return {
                'shape': spectrum.shape,
                'energy_range': (spectrum[0,0], spectrum[-1,0]),
                'max_intensity': spectrum[:,1].max(),
                'total_intensity': np.trapz(spectrum[:,1], spectrum[:,0]),
                'energy_spacing': energy_diffs[0] if np.allclose(energy_diffs, energy_diffs[0], rtol=1e-10) else 'non-uniform'
            }
            
        except Exception as e:
            print(f"ERROR loading spectrum: {e}")
            return None
    
    def plot_spectrum(self, spectrum):
        """Plot the X-ray energy spectrum"""
        plt.figure(figsize=(10, 6))
        plt.plot(spectrum[:,0], spectrum[:,1], 'b-', linewidth=2, label='X-ray Spectrum')
        plt.xlabel('Energy (keV)')
        plt.ylabel('Relative Intensity')
        plt.title('X-ray Energy Spectrum (120 kVp)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add some statistics as text
        peak_energy = spectrum[np.argmax(spectrum[:,1]), 0]
        plt.axvline(peak_energy, color='r', linestyle='--', alpha=0.7, label=f'Peak: {peak_energy:.1f} keV')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('spectrum_verification.png', dpi=150, bbox_inches='tight')
        print("✓ Spectrum plot saved as 'spectrum_verification.png'")
        plt.show()
    
    def create_debug_config(self):
        """Create a modified config for debugging"""
        debug_config = self.config.copy()
        debug_config['train']['epoch'] = 100  # Fewer epochs for debugging
        debug_config['train']['save_epoch'] = 50
        debug_config['debug'] = {
            'save_iterations': True,
            'log_p_values': True,
            'save_projections': True
        }
        
        with open('./Polyner/config_debug.json', 'w') as f:
            json.dump(debug_config, f, indent=2)
        
        print("✓ Created debug config at './Polyner/config_debug.json'")
        return debug_config
    
    def run_full_analysis(self, img_id=0):
        """Run complete debugging analysis"""
        print("🔍 Starting full Polyner debugging analysis...")
        
        # 1. Check data consistency
        files_info = self.check_data_consistency(img_id)
        
        # 2. Validate mask
        mask_info = self.validate_mask(img_id, visualize=True)
        
        # 3. Verify spectrum
        spectrum_info = self.verify_spectrum(plot=True)
        
        # 4. Create debug config
        debug_config = self.create_debug_config()
        
        # 5. Summary
        print(f"\n=== DEBUGGING SUMMARY ===")
        print("Data consistency:", "✓ PASS" if files_info else "❌ FAIL")
        print("Mask validation:", "✓ PASS" if mask_info and mask_info['is_binary'] and mask_info['metal_pixels'] > 0 else "⚠️  ISSUES FOUND")
        print("Spectrum verification:", "✓ PASS" if spectrum_info and spectrum_info['max_intensity'] > 0 else "❌ FAIL")
        
        return {
            'files_info': files_info,
            'mask_info': mask_info,
            'spectrum_info': spectrum_info
        }

if __name__ == "__main__":
    # Run debugging
    debugger = PolynerDebugger()
    results = debugger.run_full_analysis(img_id=0)