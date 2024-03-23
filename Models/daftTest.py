import pickle
import os
import sys
sys.path.append('/home/sergiu.cociuba/BoneLab')
import torch
from collections import OrderedDict
from typing import Any, Dict, Optional, Sequence
import torch.nn as nn
from DAFT.daft.models.base import BaseModel
from DAFT.daft.networks.vol_blocks import ConvBnReLU, DAFTBlock, FilmBlock, ResBlock
from DAFT.daft.networks.vol_networks import DAFT
import torch.nn.functional as F
from monai.transforms import EnsureChannelFirst, Compose, ScaleIntensity
from monai.data import Dataset, DataLoader
import logging
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import monai
from monai.data import Dataset, DataLoader
from monai.transforms import EnsureChannelFirst, Compose, ScaleIntensity
from sklearn.model_selection import KFold
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import pandas as pd
from medcam import medcam


class TensorDataset(Dataset):
    def __init__(self, data_dir, bone_type, transform=None, downsample_2=(2,2,2), tabular_transform=None):
        self.data_dir = data_dir
        self.bone_type = bone_type
        self.transform = transform
        self.downsample_2 = downsample_2
        self.tabular_transform = tabular_transform
        self.tensor_filenames = os.listdir(data_dir)

    def __len__(self):
        return len(self.tensor_filenames)

    def __getitem__(self, index):
        tensor_filename = self.tensor_filenames[index]
        tensor_path = os.path.join(self.data_dir, tensor_filename)
        tensor, tabular_row, _ = load(tensor_path) 

        tabular_row = tabular_row[1:]  

        tensor = torch.tensor(tensor)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        try:
            tensor = F.avg_pool3d(tensor, self.downsample_2)
        except Exception as e:
            print(f"Error during pooling operation: {e}")
            print(f"Tensor shape before pooling: {tensor.shape}")
            print(f"Downsample size: {self.downsample_2}")
            return None

        label = torch.tensor(float(tabular_row[22])) if self.bone_type == 'r' else torch.tensor(float(tabular_row[23]))
        tabular_row = tabular_row[:22] + tabular_row[24:]
        if self.transform is not None:
            tensor = self.transform(tensor)

        if isinstance(tabular_row, list):
            for i in tabular_row:
                try:
                    float(i) 
                except ValueError:
                    print(f"Error converting to float: {i} in row {tabular_row}")
            tabular_row = [float(i) if i != '' else 0.0 for i in tabular_row]

        if self.tabular_transform is not None:
            tabular_data = self.tabular_transform(tabular_row)
        else:
            tabular_data = torch.tensor(tabular_row)
        print("tensor shape is: ", tensor.shape)
        print("Label is:  ", label)
        print("Tabular row is: ", tabular_data)
        print("_ is: ", _)
        return tensor, label, tabular_data


def load(file_path):
    # Check if file exists
    if not os.path.exists(file_path):
        print("File does not exist.")
        return

    # Unpickle the file
    with open(file_path, 'rb') as f:
        tensor, tabular_data, _ = pickle.load(f)
        return tensor, tabular_data, _

def load_model(model_path):
    model = DAFT(
        in_channels=1,  
        n_outputs=1, 
        bn_momentum=0.1,
        n_basefilters=4
    )  
    model.load_state_dict(torch.load(model_path))
    model.eval()  
    return model

def evaluate_model(model, test_loader, device):
    model = medcam.inject(model, output_dir='attention_maps', layer='auto', save_maps=True)
    predictions = []
    targets = []
    with torch.no_grad():
        for data in test_loader:
            images, labels, tabular_data = data[0].to(device), data[1].to(device), data[2].to(device)
            output_dict = model(images, tabular_data)
            logits = output_dict["logits"].squeeze(1)
            predictions.extend(logits.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    mse = mean_squared_error(targets,predictions)
    return predictions, targets, mae, r2, mse

def main(test_data_dir, bone_type, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path).to(device)
    
    test_dataset_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(channel_dim=0)])
    test_dataset = TensorDataset(test_data_dir, bone_type, transform=test_dataset_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0, pin_memory=torch.cuda.is_available())

    predictions, targets, mae, r2, mse = evaluate_model(model, test_loader, device)

    results_df = pd.DataFrame({
        'Predictions': predictions,
        'Targets': targets,
        'MAE': [mae] * len(predictions),
        'R2': [r2] * len(predictions),
        'MSE':  [mse] * len(predictions)
    })

    results_df.to_csv('/home/sergiu.cociuba/BoneLab/DAFT/daft/test_results_daft.csv', index=False)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])  # Pass the test data directory, bone type, and model path as arguments
