## import library
import numpy as np
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms
from efficientnet_pytorch import EfficientNet

SIZE = 448
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
BATCH_SIZE = 32
LR = 1e-4
BASE_DIR = "/home/son/_study_group/kaggle/dogs_n_cats/"
DATA_DIR = BASE_DIR + 'catsdogs_dataset_kaggle/data/'
TRAIN_DIR = DATA_DIR + 'train/'
TEST_DIR = DATA_DIR + 'test/'
VAL_RATIO = .1
NUM_EPOCHS = 20
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 1
MODEL_NAME = 'efficientnet-b3'

class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(512),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'test': transforms.Compose([
                transforms.Resize(512),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)
    
class DogsCatsDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        if self.phase == 'test':
            img_path = os.path.join(TEST_DIR, self.file_list[idx])
        else:
            img_path = os.path.join(TRAIN_DIR, self.file_list[idx])
        img = Image.open(img_path)
        img_transformed = self.transform(img, self.phase)
        label = img_path.split('/')[-1].split('.')[0]
        label = 1 if label=='dog' else 0
        return img_transformed, label
    

def create_params_to_update(net):
    params_to_update_1 = []
    update_params_name_1 = ['_fc.weight', '_fc.bias']
    for name, param in net.named_parameters():
        if name in update_params_name_1:
            param.requires_grad = True
            params_to_update_1.append(param)
        else:
            param.requires_grad = False
    params_to_update = [{'params': params_to_update_1, 'lr': LR}]
    return params_to_update

def adjust_learning_rate(optimizer, epoch):
    lr = LR * (0.1**(epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr

def train_model(net, dataloader, criterion, optimizer, num_epoch):
    net = net.to(DEVICE)
    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch+1, num_epoch))

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()
            epoch_loss = 0.0
            epoch_corrects = 0

            for inputs, labels in tqdm(dataloader[phase]):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data).item()
            epoch_loss = epoch_loss / len(dataloader[phase].dataset)
            epoch_acc = float(epoch_corrects) / len(dataloader[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        # save checkpoint
        checkpoint = {
            'epoch': epoch+1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        filepath = 'checkpoint.pth'
        filepath = "/home/son/_study_group/kaggle/dogs_n_cats/base" + filepath
        torch.save(checkpoint, filepath)

    return net

def train(train_list, val_list):
    train_dataset = DogsCatsDataset(train_list, ImageTransform(SIZE, MEAN, STD), 'train')
    val_dataset = DogsCatsDataset(val_list, ImageTransform(SIZE, MEAN, STD), 'val')
    train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}
    
    net = EfficientNet.from_pretrained(MODEL_NAME)
    net._fc = nn.Linear(in_features=1536, out_features=2)
    params_to_update = create_params_to_update(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params_to_update, lr=LR, weight_decay=1e-4)
    net = train_model(net, dataloader_dict, criterion, optimizer, NUM_EPOCHS)
    return net

train_list = os.listdir(TRAIN_DIR)
test_list = os.listdir(TEST_DIR)

train_list, val_list = train_test_split(train_list, test_size=VAL_RATIO, random_state=SEED)

net = train(train_list, val_list)

with torch.no_grad():
    print('Predicting...')
    id_list = []
    pred_list = []
    for test_file in tqdm(test_list):
        net.eval()
        img_path = os.path.join(TEST_DIR, test_file)
        img = Image.open(img_path)
        _id = int(test_file.split('/')[-1].split('.')[0])
        transform = ImageTransform(SIZE, MEAN, STD)
        img_transformed = transform(img, phase='test')
        inputs = img_transformed.unsqueeze_(0)
        inputs = inputs.to(DEVICE)
        outputs = net(inputs)
        pred = F.softmax(outputs, dim=1)[:, 1].tolist()[0]
        id_list.append(_id)
        pred_list.append(pred)
        res = pd.DataFrame({'id': id_list, 'label': pred_list})
        res.sort_values(by='id', inplace=True)
        res.sort_values(by='id', inplace=True)
        res.to_csv('/home/son/_study_group/kaggle/dogs_n_cats/catsdogs_dataset_kaggle/data/submission.csv', index=False)
