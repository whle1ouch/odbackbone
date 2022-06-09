from turtle import forward
import torch 
import torch.nn as nn
from torchsummary import summary

class ConvBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super(ConvBlock, self).__init__()
        assert len(out_channels) == 3 
        "out_channels must be a list of 3 elements"
        channels_1, channels_2, channels_3 = out_channels
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, channels_3, 1, stride=stride),
            nn.BatchNorm2d(channels_3)
        )
        self.relu = nn.ReLU(inplace=True)
        self.m = nn.Sequential(
            nn.Conv2d(in_channels, channels_1, 1, stride),
            nn.BatchNorm2d(channels_1),
            self.relu,
            nn.Conv2d(channels_1, channels_2, kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm2d(channels_2),
            self.relu,
            nn.Conv2d(channels_2, channels_3, 1, stride=1),
            nn.BatchNorm2d(channels_3)
        )
        
    def forward(self, x):
        shortcut = self.shortcut(x)
        m = self.m(x)
        return self.relu(shortcut + m)
    
    
class Identity(nn.Module):
    
    def __init__(self, in_channels, out_channels, bias=True):
        super(Identity, self).__init__()
        assert len(out_channels) == 3 
        "out_channels must be a list of 3 elements"
        channels_1, channels_2, channels_3 = out_channels
        self.relu = nn.ReLU(inplace=True)
        self.m = nn.Sequential(
            nn.Conv2d(in_channels, channels_1, 1, bias=bias),
            nn.BatchNorm2d(channels_1),
            self.relu,
            nn.Conv2d(channels_1, channels_2, 3, padding=1, bias=bias),
            nn.BatchNorm2d(channels_2),
            self.relu,
            nn.Conv2d(channels_2, channels_3, 1, bias=bias),
            nn.BatchNorm2d(channels_3),
            self.relu
        )
        
    def forward(self, x):
        m = self.m(x)
        return self.relu(x + m)
        
    
class ResNetFPN(nn.Module):
    
    def __init__(self):
        super(ResNetFPN, self).__init__()
        relu = nn.ReLU(inplace=True)
        self.fpn1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            relu
        )
        
        self.out1 = nn.MaxPool2d(3, 2, 1)
        self.fpn2 = nn.Sequential(
            ConvBlock(64, [64, 64, 256], stride=1),
            Identity(256, [64, 64, 256]),
        )
        self.out2 = Identity(256, [64, 64, 256])
        
        self.fpn3 = nn.Sequential(
            ConvBlock(256, [128, 128, 512]),
            Identity(512, [128, 128, 512]),
            Identity(512, [128, 128, 512]),
        )
        self.out3 = Identity(512, [128, 128, 512])
        
        self.fpn4 = nn.Sequential(
            ConvBlock(512, [256, 256, 1024], stride=1),
            Identity(1024, [256, 256, 1024]),
            Identity(1024, [256, 256, 1024]),
            Identity(1024, [256, 256, 1024]),
            Identity(1024, [256, 256, 1024]),
            Identity(1024, [256, 256, 1024])
        )
        self.out4 = Identity(1024, [256, 256, 1024])
        
        self.fpn5 = nn.Sequential(
            ConvBlock(1024, [512, 512, 2048], stride=1),
            Identity(2048, [512, 512, 2048]),
        )
        self.out5 = Identity(2048, [512, 512, 2048])
        
    def forward(self, x):
        fpn1 = self.fpn1(x)
        fpn2 = self.fpn2(fpn1)
        fpn3 = self.fpn3(fpn2)
        fpn4 = self.fpn4(fpn3)
        fpn5 = self.fpn5(fpn4)
        fpn1 = self.out1(fpn1)
        fpn2 = self.out2(fpn2)
        fpn3 = self.out3(fpn3)
        fpn4 = self.out4(fpn4)
        fpn5 = self.out5(fpn5)
        return [fpn1, fpn2, fpn3, fpn4, fpn5]

    
  