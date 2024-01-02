import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from torch.nn.utils.rnn import pad_sequence


class ULSignalDataset(Dataset):
    def __init__(self, csv_path):
        print('Loading Dataset.....')
        with open(csv_path, 'rb') as file:
            database = pickle.load(file)
        print('Load successful.')
        self.raw_data = database[0:1000]
        self.preprocess_raw_data()

    def preprocess_raw_data(self):
        print("Reshaping data.....")
        tensor_dataset = [torch.tensor(sample, dtype=torch.float) for sample in self.raw_data]
        padded_dataset = pad_sequence(tensor_dataset, batch_first=True, padding_value=0)
        final_database = torch.stack([tensor.reshape((1, -1)) for tensor in padded_dataset])

        self.raw = final_database
        self.masked = final_database
        print("Done reshaping.")


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
        return self.raw.shape[2]

def get_data_loaders(data_path, splits, batch_size):

    dataset = ULSignalDataset(data_path)
    
    num_features = dataset.get_num_features()

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, splits)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                          shuffle=True, num_workers=1)

    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=True, num_workers=1)

    test_loader = DataLoader(test_set, batch_size=batch_size,
                            shuffle=True, num_workers=1)
    
    return train_loader, val_loader, test_loader, num_features