import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    def __init__(self):
        # Number of channels in the training images. For color images this is 3
        nc = 3
        # Size of z latent vector (i.e. size of generator input)
        nz = 100
        # Size of feature maps in generator
        ngf = 64
        # Size of feature maps in discriminator
        ndf = 64
        self.arch = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),           
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),  
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),  
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()  
        )
    
    def forward(self, input):
        return self.arch(input)