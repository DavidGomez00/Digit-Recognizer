from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision.transforms import v2
import lightning as L
from sklearn.model_selection import train_test_split

import torchvision
import os


class CustomDataset(Dataset):
    '''Custom dataset to load the digits from the
    Kaggle competition.'''

    def __init__(self, data:pd.DataFrame, transform=None):
        super(CustomDataset, self).__init__()
        self.data=data
        self.transform=transform


    def __getitem__(self, index):
        # Extract data and label
        label = self.data.iloc[index]["label"]
        data = self.data.iloc[index].drop("label")
        # Transform data into a image
        image = torch.Tensor(data.values).view(-1, 28)
        label = torch.Tensor(label)
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


    def setup(self):
        # Load the CSV file
        full_data = pd.read_csv(self.csv_path)
        # Create train and val transforms
        train_transform = v2.Compose([
            v2.RandomRotation(degrees=(-15, 15))
        ])
        # Split the dataset into train and validation sets
        train_data, val_data = train_test_split(full_data, test_size=self.val_size, random_state=self.split_seed)
        # Create datasets
        self.train_dataset = CustomDataset(train_data, transform=train_transform)
        self.val_dataset = CustomDataset(val_data)

    
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


def test():
    # Create a DataSet
    data = pd.read_csv("data/train.csv")
    myDataset = CustomDataset(data=data)
    # Get first item
    first_image, label = myDataset.__getitem__(0)
    assert isinstance(first_image, torch.Tensor)
    print(first_image.shape)
    


if __name__ == "__main__":
    test()