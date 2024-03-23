import os
import csv
import torch
from monai.transforms.intensity.array import ScaleIntensityRange
import pickle
import numpy as np

def get_file_paths(directory):
    """
    Method for obtaining the filepaths of all files in a given directory

    Parameters
    ----------
    directory (str)
        path to a directory

    Returns
    -------
    file_paths [] 
        A list containing strings of filepaths coressponding to each file present in input directory
    """
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)				
    return file_paths

def import_csv(csv_file_path):
    """
    Imports the contents of a CSV file and returns them as a list of rows.
    
    Parameters
    ----------	
    csv_file_path (str): Path to the CSV file.
    
    Returns
    -------
    list of lists: Each inner list represents a row in the CSV file.
    """
    rows = []
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            rows.append(row)
    return rows

def normalize_pixels(array):
    """
    Normalizes the pixels using Monai's ScaleIntensityRanged function
    
    Parameters
    ----------    
    normalize_pixels (numpy array): numpy array you wish to normalize
    
    Returns
    -------
    tensor: Returns a tensor that has had its pixels normalized
    """
        
    a_min = -400
    a_max = 1400
    b_min = -1
    b_max = 1
    
    scaler = ScaleIntensityRange(a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=False)
    normalized_array = scaler(array)
    return normalized_array

def pad_to_desired_shape(image):
    '''
    Pads the tensors to a specified shape

    Parameters
    ----------
    pad_to_desired_shape(tensor) : tensor you wish to pad

    Returns
    -------
    torch.tensor(padded_image): returns a torch tensor that has been padded
    '''
    desired_shape = (800, 700, 168)
    
    pad_width = []
    for i in range(len(desired_shape)):
        width = max(desired_shape[i] - image.shape[i], 0)
        pad_width.append((int(width // 2), int(width - (width // 2))))

    padded_image = np.pad(image, pad_width, mode='constant')
    padded_image = torch.tensor(padded_image)
    return padded_image

def pickle_object_to_file(obj, directory, filename):
    """
    Pickles an object and saves it to a file in a specified directory.

    Args:
        obj: The object to pickle.
        directory: The directory path where the pickled file will be saved.
        filename: The name of the file to save the pickled object.
    """
    # Create the full file path by joining the directory and filename
    file_path = os.path.join(directory, filename)
    
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)


