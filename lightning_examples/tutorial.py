import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning import Trainer


# Hyper-parameters
input_size = 784 # 28x28
hidden_size = 500
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# Fullly connected neural network with one hidden layer
class LitNeuralNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LitNeuralNet, self).__init__()
        self.input_size= input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # No activation and softmax at the end
        return out

    def training_step(self, batch, batch_idx):
        images, labels = batch
        images = images.reshape(-1, 28 * 28)

        # Forward Pass
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        
        tensorboard_logs = {'train_loss': loss}
        # use key 'log'
        return {'loss': loss, 'log': tensorboard_logs}

    def train_dataloader(self):
        # MNIST dataset
        train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
        
        # Data loader
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers = 4, shuffle = False)
        
        return train_loader
    
    def val_dataloader(self):
        test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
        
        # Data loader
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers = 4, shuffle = False)
        
        return test_loader
        

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.reshape(-1, 28 * 28)
        
        # Forward pass
        outputs = self(images)
        
        loss = F.cross_entropy(outputs, labels)
        return {"val_loss": loss}
    
    def validation_epoch_end(self, outputs):
        # outputs = list of dictionaries
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        # use key 'log'
        return {'val_loss': avg_loss, 'log': tensorboard_logs}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)
    

if __name__ == '__main__':
    model = LitNeuralNet(input_size, hidden_size, num_classes)
    
    trainer = Trainer(max_epochs=num_epochs)
    trainer.fit(model)
