import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py

class ULSignalDataset(Dataset):
    def __init__(self, csv_path, elements_in_group, mask_p):
        with h5py.File(csv_path, "r") as f:
            raw_data = np.array(f['tracings'])
            self.raw_data = raw_data[:, :, 0]
        
        self.preprocess_raw_data(elements_in_group, mask_p)

    def preprocess_raw_data(self, elements_in_group, mask_p):

        num_groups = int(4096 / elements_in_group)

        masked_data = self.raw_data.copy()

        # for i, sample in enumerate(self.raw_data):
        #     groups_to_mask = np.random.choice([0, 1], size=num_groups, p=[mask_p, 1 - mask_p])

        #     for j in range(num_groups):
        #         if groups_to_mask[j] == 0:
        #             start_idx = j * elements_in_group
        #             end_idx = (j + 1) * elements_in_group
        #             masked_data[i, start_idx:end_idx] = 0
        
        self.raw = torch.tensor(self.raw_data, dtype=torch.float)
        self.masked = torch.tensor(masked_data, dtype=torch.float)
    
    def get_num_features(self):
        return self.raw.shape[1]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        masked = self.masked[idx]
        raw = self.raw[idx]

        return masked, raw 
    
    def __len__(self):
        return self.raw.shape[0]
    
    def get_num_features(self):
        return self.raw.shape[1]

def get_data_loaders(data_path, splits, batch_size, elements_in_group, mask_p):

    dataset = ULSignalDataset(data_path, elements_in_group, mask_p)
    
    num_features = dataset.get_num_features()

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, splits)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                          shuffle=True, num_workers=1)

    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=True, num_workers=1)

    test_loader = DataLoader(test_set, batch_size=batch_size,
                            shuffle=True, num_workers=1)
    
    return train_loader, val_loader, test_loader, num_features