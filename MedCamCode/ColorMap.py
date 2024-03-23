import argparse
import nibabel as nib
import numpy as np

def main():
    """
    Main function to process a Nifti file for intensity adjustment based on a gradient with 10 divisions.
    """

    # Parse command line arguments for Nifti file input
    parser = argparse.ArgumentParser(description="Adjust the intensity of pixels in a Nifti file based on a gradient with 10 divisions.")
    parser.add_argument("nifti_file", type=str, help="Path to the Nifti file")
    args = parser.parse_args()

    # Load the Nifti file and convert it into a numpy array
    nifti_data = nib.load(args.nifti_file)
    nifti_array = nifti_data.get_fdata()

    # Define the intensity levels based on the max value
    max_value = np.max(nifti_array)
    # Create 10 divisions with a 10% increment for each
    intensity_levels = [0.1 * max_value * (i+1) for i in range(10)]

    # Adjust pixels based on the defined intensity levels
    for i in range(10):
        if i == 0:
            nifti_array[(nifti_array <= intensity_levels[i])] = i * (255/9)
        else:
            nifti_array[(nifti_array > intensity_levels[i-1]) & (nifti_array <= intensity_levels[i])] = i * (255/9)

    # Ensure all values are assigned correctly for the highest range
    nifti_array[nifti_array > intensity_levels[-2]] = 255

    # Convert the numpy array back into a Nifti file
    adjusted_nifti = nib.Nifti1Image(nifti_array, affine=nifti_data.affine)

    # Save the adjusted Nifti file
    output_filename = args.nifti_file.replace('.nii', '_10divisions_adjusted.nii')
    nib.save(adjusted_nifti, output_filename)
    print(f"Adjusted Nifti file with 10 divisions saved as {output_filename}")

if __name__ == "__main__":
    main()
