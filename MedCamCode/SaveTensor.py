import os
import pickle
import torch
import nibabel as nib
import argparse
import numpy as np

def load_and_save_nifti(file_path, output_nifti_path):
    """
    Loads a tensor from a pickle file, processes it, and saves it as a NIfTI file.

    Parameters:
    - file_path: Path to the pickle file containing the tensor.
    - output_nifti_path: Path where the NIfTI file will be saved.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        print("File does not exist.")
        return
    
    # Unpickle the file
    with open(file_path, 'rb') as f:
        tensor = pickle.load(f)
        
        # Ensure tensor is in PyTorch format and perform operations
        tensor = tensor[0].float()
        tensor = tensor.unsqueeze(0)  # Add a batch dimension
        tensor = torch.nn.functional.avg_pool3d(tensor, (2, 2, 2))
        tensor = tensor.squeeze(0)  # Remove the batch dimension

        print(tensor.shape)

        # Convert tensor to numpy array for NIfTI saving
        np_array = tensor.numpy()

        # Create a NIfTI image from the numpy array
        nifti_img = nib.Nifti1Image(np_array, affine=np.eye(4))

        # Save the NIfTI image to disk
        nib.save(nifti_img, output_nifti_path)
        print(f"Saved NIfTI file at: {output_nifti_path}")

def main():
    """
    Main function to handle command line arguments for converting a tensor from a pickle file to a NIfTI file.
    """
    parser = argparse.ArgumentParser(description="Load a tensor from a pickle file, process it, and save it as a NIfTI file.")
    parser.add_argument("file_path", type=str, help="Path to the input pickle file.")
    parser.add_argument("output_nifti_path", type=str, help="Path for the output NIfTI file.")

    args = parser.parse_args()

    load_and_save_nifti(args.file_path, args.output_nifti_path)

if __name__ == "__main__":
    main()
