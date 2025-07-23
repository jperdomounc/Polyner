import numpy as np
import nrrd
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os

def convert_nrrd_to_nii(nrrd_path, output_path):
    """Convert NRRD file to NII format"""
    try:
        # Load the nrrd file
        data, header = nrrd.read(nrrd_path)
        print(f"Successfully loaded NRRD file")
        print(f"Data shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        
        # Create a NIfTI image
        nii_img = nib.Nifti1Image(data, affine=np.eye(4))
        
        # Save as NII file
        nib.save(nii_img, output_path)
        print(f"Successfully converted to NII format: {output_path}")
        
        return data, header
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None, None

def display_nii_volume(data):
    """Display NII volume with interactive slider"""
    if data is None:
        print("No data to display")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.2)
    
    # Initial slice to display
    slice_idx = data.shape[2] // 2 if len(data.shape) >= 3 else 0
    
    # Display the slice
    if len(data.shape) >= 3:
        im = ax.imshow(data[:, :, slice_idx], cmap='gray', origin='lower')
        ax.set_title(f'Slice {slice_idx} of {data.shape[2]}')
        
        # Create slider
        ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
        slider = Slider(ax_slider, 'Slice', 0, data.shape[2]-1, valinit=slice_idx, valfmt='%d')
        
        def update(val):
            slice_idx = int(slider.val)
            im.set_array(data[:, :, slice_idx])
            ax.set_title(f'Slice {slice_idx} of {data.shape[2]}')
            fig.canvas.draw()
        
        slider.on_changed(update)
        
    else:
        im = ax.imshow(data, cmap='gray', origin='lower')
        ax.set_title('2D Image')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax)
    plt.show()

if __name__ == "__main__":
    # File paths
    nrrd_path = "/Users/juanperdomo/Desktop/PolynerCode/Polyner/input/ma_0"
    output_path = "/Users/juanperdomo/Desktop/PolynerCode/Polyner/input/ma_0.nii"
    
    # Convert NRRD to NII
    data, header = convert_nrrd_to_nii(nrrd_path, output_path)
    
    # Display the volume
    if data is not None:
        print(f"Data statistics:")
        print(f"  Min: {np.min(data)}")
        print(f"  Max: {np.max(data)}")
        print(f"  Mean: {np.mean(data)}")
        print(f"  Std: {np.std(data)}")
        
        display_nii_volume(data)