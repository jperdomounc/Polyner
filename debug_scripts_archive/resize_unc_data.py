import SimpleITK as sitk
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import os

def resize_image_to_256(input_path, output_path):
    """Resize image from 216x216 to 256x256 to match original repository format"""
    
    # Load with SimpleITK
    sitk_img = sitk.ReadImage(input_path)
    data = sitk.GetArrayFromImage(sitk_img)
    
    print(f"Original shape: {data.shape}")
    
    if data.shape == (216, 216):
        # Calculate zoom factors to resize to 256x256
        zoom_factor = 256 / 216
        
        # Resize using cubic interpolation for better quality
        resized_data = zoom(data, zoom_factor, order=3, mode='constant', cval=0)
        
        # Ensure exact 256x256 size (zoom might give 255 or 257 due to rounding)
        if resized_data.shape != (256, 256):
            from skimage.transform import resize
            resized_data = resize(resized_data, (256, 256), order=3, preserve_range=True, anti_aliasing=True)
        
        resized_data = resized_data.astype(data.dtype)
        print(f"Resized shape: {resized_data.shape}")
        print(f"Original range: [{data.min():.6f}, {data.max():.6f}]")
        print(f"Resized range: [{resized_data.min():.6f}, {resized_data.max():.6f}]")
        
        # Create new SimpleITK image with adjusted spacing
        output_img = sitk.GetImageFromArray(resized_data)
        
        # Adjust spacing to maintain physical size
        original_spacing = sitk_img.GetSpacing()
        new_spacing = [original_spacing[0] * 216/256, original_spacing[1] * 216/256]
        output_img.SetSpacing(new_spacing)
        output_img.SetOrigin(sitk_img.GetOrigin())
        
        # Save
        sitk.WriteImage(output_img, output_path)
        print(f"✓ Saved resized image to {output_path}")
        
        return resized_data
    else:
        print(f"⚠️  Unexpected shape: {data.shape}")
        return None

def resize_sinogram(input_path, output_path, target_width=611):
    """Resize sinogram to match original format"""
    
    sitk_img = sitk.ReadImage(input_path)
    data = sitk.GetArrayFromImage(sitk_img)
    
    print(f"Original sinogram shape: {data.shape}")
    
    if data.shape[0] == 360:  # Keep height, resize width
        zoom_factor = (1.0, target_width / data.shape[1])
        resized_data = zoom(data, zoom_factor, order=3, mode='constant', cval=0)
        
        # Ensure exact dimensions
        if resized_data.shape[1] != target_width:
            from skimage.transform import resize
            resized_data = resize(resized_data, (360, target_width), order=3, preserve_range=True, anti_aliasing=True)
        
        resized_data = resized_data.astype(data.dtype)
        print(f"Resized sinogram shape: {resized_data.shape}")
        
        output_img = sitk.GetImageFromArray(resized_data)
        output_img.SetSpacing([1.0, 1.0])
        output_img.SetOrigin([0.0, 0.0])
        
        sitk.WriteImage(output_img, output_path)
        print(f"✓ Saved resized sinogram to {output_path}")
        
        return resized_data
    else:
        print(f"⚠️  Unexpected sinogram shape: {data.shape}")
        return None

def resize_sensor_pos(input_path, output_path):
    """Resize fan sensor position to match original format"""
    
    sitk_img = sitk.ReadImage(input_path)
    data = sitk.GetArrayFromImage(sitk_img)
    
    print(f"Original sensor shape: {data.shape}")
    
    # Transpose and resize to match (1, 611) format
    if data.shape == (520, 1):
        # First transpose to (1, 520), then resize to (1, 611)
        data_transposed = data.T  # Now (1, 520)
        
        zoom_factor = (1.0, 611 / 520)
        resized_data = zoom(data_transposed, zoom_factor, order=3, mode='constant', cval=0)
        
        # Ensure exact dimensions
        if resized_data.shape != (1, 611):
            from skimage.transform import resize
            resized_data = resize(resized_data, (1, 611), order=3, preserve_range=True, anti_aliasing=True)
            
        resized_data = resized_data.astype(data.dtype)
        print(f"Resized sensor shape: {resized_data.shape}")
        
        output_img = sitk.GetImageFromArray(resized_data)
        output_img.SetSpacing([1.0, 1.0])
        output_img.SetOrigin([0.0, 0.0])
        
        sitk.WriteImage(output_img, output_path)
        print(f"✓ Saved resized sensor position to {output_path}")
        
        return resized_data
    else:
        print(f"⚠️  Unexpected sensor shape: {data.shape}")
        return None