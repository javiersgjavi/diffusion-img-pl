import os
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
import torchvision
import pytorch_lightning as pl
from PIL import Image

from torch.utils.data import DataLoader, random_split


class CifarDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data/cifar10-64', batch_size=16):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.data_train = None
        self.data_test = None

    def setup(self, stage='train'):
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
            torchvision.transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.data_train = torchvision.datasets.ImageFolder(f'{self.data_dir}/train', transform=transforms)
        self.data_test = torchvision.datasets.ImageFolder(f'{self.data_dir}/test', transform=transforms)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False)
    
    def predict_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False)

