import sys
import os
sys.path.append('/home/sergiu.cociuba/BoneLab/DenseNet256/Preprocessing_Fix/Bonelab/MONAI/monai')
sys.path.append('/home/sergiu.cociuba/BoneLab/DenseNet256/Preprocessing_Fix/Bonelab')
sys.path.append('/home/sergiu.cociuba/BoneLab/DenseNet256/Preprocessing_Fix/bonelab_pytorch_lightning/')
from preprocessing import get_file_paths
from preprocessing import import_csv
from preprocessing import normalize_pixels
from preprocessing import pad_to_desired_shape
from preprocessing import pickle_object_to_file
from blpytorchlightning.dataset_components.file_loaders.AIMLoader import AIMLoader

# First argument passed will contain the filepath to your images
tensor_dir = sys.argv[1]

# Second argument passed will contain the filetype (pattern)
# NOTE: This should be a .AIM unless the code is being modified for another type of imaging
pattern = sys.argv[2]

# Instantiate the aim_loader
aim_loader = AIMLoader(tensor_dir,pattern)
# Third argument passed will contain the tabular data

tabular_dir = sys.argv[3]
tabular_data = import_csv(tabular_dir)

# Fourth Argument will contain the output directory for the preprocessed tensors
output_dir = sys.argv[4]

# NOTE: the DEFAULT PADDING size is (914, 881, 168). All images will end up being size (914, 881, 168). 
# To change this, change the desired_shape variable located in pad_to_desired_shape() method found in the Preprocessing.py file

for image_data, fn in aim_loader:
    # Extracting ID from image filename
    fn_modified = os.path.basename(fn).replace("_RL", "").replace("_RR", "").replace("_TR", "").replace("_TL", "").replace("_M00", "")
    basename, extension = os.path.splitext(fn_modified)
    image_id = basename.strip()

    # Find matching row in tabular dataset
    matching_tabular_row = None
    for row in tabular_data:
        print("Row ID: ", row[0].strip())
        print("Image ID: ", image_id)
        if row[0].strip() == image_id: 
            matching_tabular_row = row
            print("Match found")
            break

    if matching_tabular_row:
        print("Match found again")
        print("fn: ", fn)
        print("basename: ", image_id)
        print("Image_data: ", image_data.shape)
        normalized_image_data = normalize_pixels(image_data)
        print("Normalized: ", normalized_image_data.shape)
        padded_tensor = pad_to_desired_shape(normalized_image_data)
        print("Padded: ", padded_tensor.shape)

        # Save tuple including the tabular row
        save_tuple = (padded_tensor, matching_tabular_row, fn)
        print("Saving Tuples")
        filename = os.path.basename(fn)
        basename, extension = os.path.splitext(filename)
        pickle_object_to_file(save_tuple, output_dir, basename)
