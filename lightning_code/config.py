import torch.nn as nn
import torchvision.transforms as transforms

TRAIN_DATA_DIR = "/home/son/_study_group/kaggle/dogs_n_cats/catsdogs_dataset_kaggle/data/train"
PRED_DATA_DIR = "/home/son/_study_group/kaggle/dogs_n_cats/catsdogs_dataset_kaggle/data/test"
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_WORKERS = 2
PRECISION = 16
NUM_EPOCHS = 5
DEVICES = 1

LOSS_FN = nn.CrossEntropyLoss()

TRANSFORM = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=(224, 224)), 
            # transforms.CenterCrop(448),
            transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
        ])
TRANSFORM_PRED = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(size=(448, 448)), 
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
        ])



