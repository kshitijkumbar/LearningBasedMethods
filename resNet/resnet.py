import torch
import numpy as np
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super(ResBlock,self).__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, 
                                    out_channels, 
                                    kernel_size=3,
                                    stride=2,
                                    padding=1)
            self.shortcut = nn.Sequential(
                            nn.Conv2d(in_channels, 
                                    out_channels, 
                                    kernel_size=1,
                                    stride=2),
                            nn.BatchNorm2d(out_channels)
                            )
        else:
            self.conv1 = nn.Conv2d(in_channels, 
                                    out_channels, 
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
            self.shortcut = nn.Sequential()
        
        self.conv2 = nn.Conv2s(out_channels, out_channels, kernel_size=3,
                            stride=1, padding=1)

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = input + shortcut
        return nn.ReLU()(input)


class ResNet18(nn.module):
    def __init__(self, in_channels, resblock, outputs=1000):
        super().__init__()
        self.layer0 = nn.Sequential(
                        nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU()
                        )
        
        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False)
        )


        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False)
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(512, outputs)

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = torch.flatten(input)
        input = self.fc(input)

        return input        

            