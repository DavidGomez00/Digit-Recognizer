import lightning as L
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score
from torch.optim import SGD


class ConvBlock(nn.Module):
    '''Block of convolutions, 3 convolutions with
    a kernel 3x3.'''

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels

    def forward(self, x):
        x = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(3,3), padding=1)
        x = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=(3,3), padding=1)
        x = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=(3,3), padding=1)
        return x


class CustomModel(nn.Module):
    '''CNN for 10 class classification'''

    def __init__(self):
        super(CustomModel, self).__init__()
        # Define convolutions
        self.convBlock1 = ConvBlock(in_channels=3, out_channels=16)
        self.convBlock2 = ConvBlock(in_channels=16, out_channels=32)
        self.convBlock3 = ConvBlock(in_channels=32, out_channels=64)
        # Pool and batchnorm
        self.pool = nn.MaxPool2d(kernel_size=(2,2))
        self.batch_norm = nn.BatchNorm2d()
        # Linear layers
        self.ff1 = nn.Linear(in_features=7*7*64, out_features=10)

    def forward(self, x):
        # Feature extraction
        x = F.relu(self.convBlock1(x))
        x = self.batch_norm(x)
        x = self.pool(x)
        x = F.relu(self.convBlock2(x))
        x = self.batch_norm(x)
        x = self.pool(x)
        x = F.relu(self.convBlock3(x))
        x = self.batch_norm(x)
        x = nn.Flatten(x)
        # Classification
        x = self.ff1(x)
        return x


class CustomLighningModule(L.LightningModule):
    '''Lighning wrapper for the model optimizer, 
    train/val steps and logging.'''

    def __init__(self, learning_rate):
        super(CustomLighningModule, self).__init__()
        # loss function
        self.loss_fn = nn.BCEWithLogitsLoss()
        # Metrics
        self.train_outputs = []
        self.val_outputs = []
        self.acc_score = Accuracy('multiclass', num_classes=10)
        self.f1_score = F1Score('multiclass', num_classes= 10)
        # Model
        self.model = CustomModel()
        self.learning_rate = learning_rate

    
    def training_step(self, batch, batch_idx):
        # Compute loss and scores
        loss, scores, y = self._common_step(batch, batch_idx)
        # Log
        self.log_dict({'train_loss': loss}, on_step=False, on_epoch=True, prog_bar=True)
        self.train_outputs.append({'scores': scores, 'y':y})
        return {'loss': loss, 'scores': scores, 'y':y}
    

    def on_train_epoch_end(self):
        # Retrieve scores and labels and clear the outputs for the epoch
        scores = torch.cat([x['scores'] for x in self.train_outputs])
        y = torch.cat([x['y'] for x in self.train_outputs])
        self.train_outputs = []
        # Compute metrics for the epoch and log
        self.log_dict({
            'train_accuracy': self.acc_score(scores, y),
            'train_f1': self.f1_score(scores, y)
        },
        on_step=False,
        on_epoch=True,
        prog_bar=True
        )


    def validation_step(self, batch, batch_idx):
        # Compute loss and scores
        loss, scores, y = self._common_step(batch, batch_idx)
        # Log
        self.log_dict({'val_loss': loss}, on_step=False, on_epoch=True, prog_bar=True)
        self.val_outputs.appen({'scores':scores, 'y':y})
        return {'loss':loss, 'scores':scores, 'y':y}
    

    def on_validation_epoch_end(self):
        # Retrieve scores and labels and clear the outputs for the epoch
        scores = torch.cat(x['scores'] for x in self.val_outputs)
        y = torch.cat(x['y'] for x in self.val_outputs)
        self.val_outputs = []
        # Compute metrics for the epoch and log
        self.log_dict({
            'val_accuracy':self.acc_score(scores, y),
            'val_f1':self.f1_score(scores, y)
        },
        on_step=False,
        on_epoch=False,
        prog_bar=True
        )
    

    def _common_step(self, batch, batch_idx):
        '''Perform common computations for each step'''
        x, y = batch
        scores = self.model(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y


    def configure_optimizers(self):
        return SGD(parameters=self.model.parameters(), lr=self.learning_rate)


        