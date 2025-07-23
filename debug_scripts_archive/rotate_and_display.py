import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def load_nii_file(file_path):
    """Load NII file and return data"""
    try:
        nii_img = nib.load(file_path)
        data = nii_img.get_fdata()
        print(f"Loaded NII file: {file_path}")
        print(f"Data shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        return data
    except Exception as e:
        print(f"Error loading NII file: {e}")
        return None

def apply_rotations(data):
    """Apply different rotation options to the data"""
    if data is None:
        return None
    
    # Since it's a 3D volume with shape (216, 216, 1), we'll work with the 2D slice
    if len(data.shape) == 3 and data.shape[2] == 1:
        slice_2d = data[:, :, 0]
    else:
        slice_2d = data
    
    rotations = {
        'Original': slice_2d,
        'Rotate 90°': np.rot90(slice_2d, k=1),
        'Rotate 180°': np.rot90(slice_2d, k=2),
        'Rotate 270°': np.rot90(slice_2d, k=3),
        'Flip Horizontal': np.fliplr(slice_2d),
        'Flip Vertical': np.flipud(slice_2d),
        'Transpose': np.transpose(slice_2d),
        'Rotate 90° + Flip H': np.fliplr(np.rot90(slice_2d, k=1))
    }
    
    return rotations

def display_rotations(rotations):
    """Display all rotation options in a grid"""
    if rotations is None:
        print("No rotations to display")
        return
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, (name, data) in enumerate(rotations.items()):
        ax = axes[i]
        im = ax.imshow(data, cmap='gray', origin='lower')
        ax.set_title(name)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Add data statistics
        ax.text(0.02, 0.98, f'Min: {np.min(data):.3f}\nMax: {np.max(data):.3f}', 
                transform=ax.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.suptitle('Different Rotation Options for ma_0.nii', fontsize=16, y=1.02)
    plt.show()

def save_rotated_version(data, rotation_name, output_path):
    """Save a specific rotation as a new NII file"""
    try:
        # Create a new NIfTI image with the rotated data
        if len(data.shape) == 2:
            # Add back the third dimension if it was a 2D slice
            data_3d = np.expand_dims(data, axis=2)
        else:
            data_3d = data
            
        nii_img = nib.Nifti1Image(data_3d, affine=np.eye(4))
        nib.save(nii_img, output_path)
        print(f"Saved {rotation_name} version to: {output_path}")
    except Exception as e:
        print(f"Error saving rotated version: {e}")

if __name__ == "__main__":
    # Load the NII file
    nii_path = "/Users/juanperdomo/Desktop/PolynerCode/Polyner/input/ma_0.nii"
    data = load_nii_file(nii_path)
    
    if data is not None:
        # Apply different rotations
        rotations = apply_rotations(data)
        
        # Display all rotation options
        display_rotations(rotations)
        
        # Print instructions for user
        print("\nRotation options displayed:")
        print("1. Original - No rotation")
        print("2. Rotate 90° - Counterclockwise 90 degrees")
        print("3. Rotate 180° - 180 degrees")
        print("4. Rotate 270° - Counterclockwise 270 degrees (or clockwise 90°)")
        print("5. Flip Horizontal - Mirror horizontally")
        print("6. Flip Vertical - Mirror vertically")
        print("7. Transpose - Swap X and Y axes")
        print("8. Rotate 90° + Flip H - Combined rotation and flip")
        
        # Ask user which rotation they prefer (this would be interactive in a real scenario)
        print("\nTo save a specific rotation, modify the script to specify which one you want.")
        
        # Example: Save the 90-degree rotation
        # save_rotated_version(rotations['Rotate 90°'], 'Rotate 90°', 
        #                     "/Users/juanperdomo/Desktop/PolynerCode/Polyner/input/ma_0_rotated90.nii")