import os
import torch
import pytorch_lightning as pl
from PIL import Image
import numpy as np

class MyDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, predict_data_dir, batch_size, num_workers, transform):
        super().__init__()
        self.data_dir = data_dir
        self.predict_data_dir = predict_data_dir
        self.transform = transform
        self.batchsize = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        entire_dataset = MyDataset(self.data_dir, self.transform)
        train_set_size = int(len(entire_dataset)*.8)
        val_set_size = int(len(entire_dataset)*.1)
        test_set_size = int(len(entire_dataset)*.1)
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(entire_dataset, [train_set_size, val_set_size, test_set_size])
        self.pred_dataset = MyDataset(self.predict_data_dir, self.transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batchsize, shuffle=True, 
                                            num_workers = self.num_workers, pin_memory = True, persistent_workers = True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batchsize, shuffle=False, 
                                            num_workers = self.num_workers, pin_memory = True, persistent_workers = True)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batchsize, shuffle=False, 
                                            num_workers = self.num_workers, pin_memory = True, persistent_workers = True)
    
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.pred_dataset, batch_size=self.batchsize, shuffle=False, 
                                            num_workers = self.num_workers, pin_memory = True, persistent_workers = True)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform = None):
        super(MyDataset, self).__init__()
        self.data_dir = data_dir
        self.images_list = os.listdir(self.data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        image_dir = self.images_list[index]
        img = Image.open(os.path.join(self.data_dir, image_dir))
        img = np.array(img)
        if self.transform != None:
            img = self.transform(img)

        # train folder has ground truth, but test folder doesnt
        if "train" in self.data_dir:
            if "cat" in image_dir: label = 1
            elif "dog" in image_dir: label = 0
            else: label = -1
            return img, label
        else:
            return img