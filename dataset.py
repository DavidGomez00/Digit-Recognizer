from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchvision.transforms as transforms
import lightning as L
from sklearn.model_selection import train_test_split

import numpy as np
import os


class CustomDataset(Dataset):
    '''Custom dataset to load the digits from the
    Kaggle competition.'''

    def __init__(self, data:pd.DataFrame, transform=None):
        super(CustomDataset, self).__init__()
        self.data=data
        self.transform=transform


    def __getitem__(self, index):
        image = self.data.iloc[index, 1:].values.astype(np.float32).reshape((28,28))
        label = [0.0] * 10
        label[self.data.iloc[index, 0]] = 1.0
        label = torch.Tensor(label).to(torch.float32)

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        return image, label
    

    def __len__(self):
        return len(self.data)


class CustomDataModule(L.LightningDataModule):
    '''Wrapper for the dataloaders and dataset'''

    def __init__(self, csv_path:os.PathLike, batch_size:int, num_workers:int, val_size:float, split_seed:int):
        super(CustomDataModule, self).__init__()
        # Dataloader parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Data
        self.csv_path=csv_path
        self.val_size=val_size
        self.train_dataset = None
        self.val_dataset = None
        # Config
        self.split_seed=split_seed


    def setup(self, stage):
        # Load the CSV file
        full_data = pd.read_csv(self.csv_path)

        # Create train and val transforms
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=(-15, 15)),
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # Split the dataset into train and validation sets
        train_data, val_data = train_test_split(full_data, test_size=self.val_size, random_state=self.split_seed)
        # Create datasets
        self.train_dataset = CustomDataset(train_data, transform=train_transform)
        self.val_dataset = CustomDataset(val_data, transform=valid_transform)

    
    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)
    

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)
