import numpy as np
from scipy.ndimage import zoom
import nibabel as nib
import argparse

def load_nifti_file(file_path):
    """
    Load a NIfTI file and return its data and affine matrix.
    
    Parameters:
    - file_path (str): Path to the NIfTI file.
    
    Returns:
    - tuple: A tuple containing:
        - data (numpy.ndarray): Numpy array containing the image data.
        - affine (numpy.ndarray): Affine matrix for the NIfTI file.
    """
    nii = nib.load(file_path)
    data = nii.get_fdata()
    affine = nii.affine
    return data, affine

def upscale_attention_map(attention_map, target_shape):
    """
    Upscale an attention map to a target shape using zoom.
    
    Parameters:
    - attention_map (numpy.ndarray): The attention map to be upscaled.
    - target_shape (tuple): The target dimensions to upscale to.
    
    Returns:
    - numpy.ndarray: The attention map upscaled to the target shape.
    """
    print(target_shape)
    zoom_factors = np.array(target_shape) / np.array(attention_map.shape)
    upscaled_attention_map = zoom(attention_map, zoom_factors, order=1)
    return upscaled_attention_map

def save_nifti(image_data, affine, save_path):
    """
    Save image data as a NIfTI file.
    
    Parameters:
    - image_data (numpy.ndarray): The image data to save.
    - affine (numpy.ndarray): The affine matrix for the NIfTI image.
    - save_path (str): The path to save the NIfTI file at.
    """
    nifti_img = nib.Nifti1Image(image_data, affine)
    nib.save(nifti_img, save_path)
    print(f"Saved NIfTI file at: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upscale an attention map and save as a NIfTI file.")
    parser.add_argument("--p1", type=str, required=True, help="Path to the original NIfTI image.")
    parser.add_argument("--p2", type=str, required=True, help="Path to the attention map NIfTI file.")
    parser.add_argument("--p3", type=str, required=True, help="Path to save the upscaled NIfTI file.")
    
    args = parser.parse_args()
    
    # Load original image and attention map
    original_image, original_affine = load_nifti_file(args.p1)
    attention_map, _ = load_nifti_file(args.p2)
    
    # Upscale attention map
    upscaled_attention_map = upscale_attention_map(attention_map, original_image.shape)
    
    # Save the NIfTI file
    save_nifti(upscaled_attention_map, original_affine, args.p3)
