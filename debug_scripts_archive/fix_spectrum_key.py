import scipy.io as scio
import numpy as np

def fix_spectrum_key(input_path, output_path):
    """Fix the spectrum file key from 'HE_MAR' to 'GE14Spectrum120KVP'"""
    
    # Load your current spectrum file
    data = scio.loadmat(input_path)
    
    print(f"Current keys: {[k for k in data.keys() if not k.startswith('__')]}")
    
    # Extract the spectrum data from 'HE_MAR' key
    spectrum_data = data['HE_MAR']
    
    print(f"Spectrum data shape: {spectrum_data.shape}")
    print(f"Energy range: {spectrum_data[0,0]} to {spectrum_data[-1,0]} keV")
    print(f"Max intensity: {np.max(spectrum_data[:,1])}")
    
    # Create new dictionary with the correct key name
    corrected_data = {
        'GE14Spectrum120KVP': spectrum_data
    }
    
    # Save with the correct key
    scio.savemat(output_path, corrected_data)
    print(f"✓ Saved corrected spectrum file to {output_path}")
    
    # Verify the fix
    verification = scio.loadmat(output_path)
    print(f"Verification - new keys: {[k for k in verification.keys() if not k.startswith('__')]}")

if __name__ == "__main__":
    input_file = "/Users/juanperdomo/Desktop/PolynerCode/Polyner/input/GE14Spectrum120KVP.mat"
    output_file = input_file  # Overwrite the same file
    
    fix_spectrum_key(input_file, output_file)