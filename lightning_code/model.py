import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet
from torchsummary import summary
import torch.nn.functional as F

class Net(pl.LightningModule):
    def __init__(self, lr, loss_fn):
        super().__init__()
        self.save_hyperparameters()

        self.net = torchvision.models.vgg16(pretrained=True)
        self.net.classifier[6] = nn.Linear(in_features=self.net.classifier[6].in_features, out_features=2)
        params_to_update = []
        update_params_name = ['classifier.6.weight', 'classifier.6.bias']
        for name, param in self.net.named_parameters():
            if name in update_params_name:
                param.requires_grad = True
                params_to_update.append(param)
            else:
                param.requires_grad = False
        # self.net = EfficientNet.from_pretrained("efficientnet-b0", num_classes=1)
        # num_ftrs = self.net._fc.in_features
        # self.net._fc = nn.Sequential(
        #     nn.Linear(num_ftrs, 1),
        #     nn.Sigmoid()
        # )
        # for param in self.net.parameters():
        #     param.requires_grad = False
        # self.net._fc = nn.Sequential(
        #     nn.Linear(1280, 1),
        #     # nn.ReLU(),
        #     # nn.Linear(10, 1),
        #     # nn.Sigmoid()
        # )
        self.lr = lr
        self.loss_fn = loss_fn

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer =  torch.optim.Adam(self.parameters(), lr = self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.1)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        # y = y.float()
        pred = self.forward(x).squeeze(1)
        loss = self.loss_fn(pred, y)

        # Calc Correct
        _, preds = torch.max(pred, 1)
        correct = torch.sum(preds==y.data)/pred.shape[0]

        self.log_dict(
            {
                "train_loss": loss, 
                "train_acc": correct,
            }, 
            on_step=True,
            on_epoch=True, 
            prog_bar=True, 
            # sync_dist=True
        )
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        # y = y.float()
        pred = self.forward(x).squeeze(1)
        loss = self.loss_fn(pred, y)

        # Calc Correct
        _, preds = torch.max(pred, 1)
        correct = torch.sum(preds==y.data)/pred.shape[0]

        self.log_dict(
            {
                "val_loss": loss, 
                "val_acc": correct,
            }, 
            on_step=False,
            on_epoch=True, 
            prog_bar=False, 
            # sync_dist=True
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        # y = y.float()
        pred = self.forward(x).squeeze(1)
        loss = self.loss_fn(pred, y)
        self.log_dict(
            {
                "test_loss": loss, 
            }, 
            on_step=False,
            on_epoch=True, 
            prog_bar=False, 
            # sync_dist=True
        )

    def predict_step(self, batch, batch_idx): # make prediction
        x = batch
        outputs = self.forward(x)
        preds = F.softmax(outputs, dim=1)[:,1]
        return preds