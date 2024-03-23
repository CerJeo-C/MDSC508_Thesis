import logging
import sys
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import monai
from monai.data import Dataset, DataLoader
from monai.transforms import EnsureChannelFirst, Compose, ScaleIntensity
from monai.networks.nets import ResNet
from sklearn.model_selection import KFold
import os
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import pickle
sys.path.append('/home/sergiu.cociuba/BoneLab')


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
        return tensor, label
    
def load(file_path):
    # Check if file exists
    if not os.path.exists(file_path):
        print("File does not exist.")
        return

    # Unpickle the file
    with open(file_path, 'rb') as f:
        tensor, tabular_data, _ = pickle.load(f)
        return tensor, tabular_data, _
        
def main(data_dir, bone_type):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print("Model is: Resnet")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(
        block='basic',  
        layers=[2, 2, 2, 2],  
        block_inplanes=[64, 128, 256, 512],  
        spatial_dims=3,  
        n_input_channels=1,  
        conv1_t_size=(7, 7, 7), 
        conv1_t_stride=(2, 2, 2),  
        no_max_pool=True,  
        shortcut_type="B", 
        widen_factor=1.0,  
        num_classes=1,  
        feed_forward=True, 
        bias_downsample=True
    )
    model.to(device)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)

    scaler = GradScaler()
    writer = SummaryWriter()
    early_stop_patience = 20
    epochs_without_improvement = 0 
    batch_size = 4
    val_interval = 2
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)

    dataset_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(channel_dim=0)])
    dataset = TensorDataset(data_dir, bone_type, transform=dataset_transforms)

    fold_results = pd.DataFrame(columns=['Fold', 'Best Val Loss', 'Best Val MAE', 'Best Val R2'])

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f"FOLD {fold}")
        print("--------------------------------")
        epochs_without_improvement = 0
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler, num_workers=3, pin_memory=torch.cuda.is_available())
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_subsampler, num_workers=3, pin_memory=torch.cuda.is_available())

        best_val_loss = np.inf
        best_val_mae = np.inf
        best_val_r2 = -np.inf

        train_records = pd.DataFrame(columns=['Epoch', 'Batch', 'Prediction', 'Loss', 'True Value'])
        val_records = pd.DataFrame(columns=['Epoch', 'Batch', 'Prediction', 'Loss', 'True Value'])

        for epoch in range(100):
            print(f"Epoch {epoch + 1}/100")
            model.train()
            epoch_loss = 0

            for batch_data in train_loader:
                inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                if epoch == 0:
                    print(f"Training labels: {labels}")
                optimizer.zero_grad()
                with autocast():
                    logits = model(inputs)
                    print(f"Logits shape before squeeze: {logits.shape}")
                    for i in range(logits.size(0)):  
                            print(logits[i].item()) 
                    logits = logits.squeeze(1)
                    print(f"Logits shape: {logits.shape}")
                    print(f"Labels shape: {labels.shape}")
                    for i in range(logits.size(0)):  
                        print(logits[i].item())  
                    loss = loss_function(logits.float(), labels.float())
                    print("This is the real loss : ", loss)
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
                train_records = train_records.append({
                    'Epoch': epoch,
                    'Prediction': logits.detach().cpu().numpy(),
                    'Loss': loss.item(),
                    'True Value': labels.cpu().numpy()
                }, ignore_index=True)

            epoch_loss /= len(train_loader)
            writer.add_scalar(f"train_loss_fold_{fold}", epoch_loss, epoch)


            if (epoch + 1) % val_interval == 0:
                model.eval()
                val_loss = 0
                val_preds = []
                val_targets = []

                with torch.no_grad():
                    for val_data in val_loader:
                        val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                        if epoch == 0:
                            print(f"Validation labels: {val_labels}")
                        logits = model(val_images)
                        logits = logits.squeeze(1)
                        print(f"Logits shape: {logits.shape}")
                        print(f"Labels shape: {val_labels.shape}")
                        for i in range(logits.size(0)):  
                            print(logits[i].item()) 
                        val_loss += loss_function(logits.float(), val_labels.float()).item()
                        print("This is the real loss : ", val_loss)
                        val_preds.extend(logits.view(-1).cpu().numpy())
                        val_targets.extend(val_labels.cpu().numpy())
                        val_records = val_records.append({
                        'Epoch': epoch,
                        'Prediction': logits.detach().cpu().numpy(),
                        'Loss': loss.item(),
                        'True Value': val_labels.cpu().numpy()
                    }, ignore_index=True)

                val_loss /= len(val_loader)
                val_mae = mean_absolute_error(val_targets, val_preds)
                val_r2 = r2_score(val_targets, val_preds)

                writer.add_scalar(f"val_loss_fold_{fold}", val_loss, epoch)
                writer.add_scalar(f"val_mae_fold_{fold}", val_mae, epoch)
                writer.add_scalar(f"val_r2_fold_{fold}", val_r2, epoch)

                print(f"Validation Loss - Fold {fold}, Epoch {epoch + 1}: {val_loss:.4f}, MAE: {val_mae:.4f}, R2: {val_r2:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_mae = val_mae
                    best_val_r2 = val_r2
                    epochs_without_improvement = 0
                    # Save the model checkpoint if it's the best so far
                    torch.save(model.state_dict(), f"Best_Model_Fold_{fold}_Epoch_{epoch}.pth")
                else:
                    epochs_without_improvement = epochs_without_improvement + 1

            if epochs_without_improvement >= early_stop_patience:
                print("Early stopping triggered for fold:", fold)
                break

        fold_results = fold_results.append({
            'Fold': fold,
            'Best Val Loss': best_val_loss,
            'Best Val MAE': best_val_mae,
            'Best Val R2': best_val_r2
        }, ignore_index=True)
        train_records.to_csv(f'train_records_fold_{fold}_epoch_{epoch}.csv', index=False)
        val_records.to_csv(f'val_records_fold_{fold}_epoch_{epoch}.csv', index=False)

        print(f"Completed Fold {fold}")

    writer.close()
    print("Training completed.")
    print("Fold Results:\n", fold_results)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])  # Pass the data directory and bone type as arguments
