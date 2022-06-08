import torch
import torch.nn as nn
from torchsummary import summary


class ConvBn(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class InceptionV2Block(nn.Module):
    
    def __init__(self, in_channels, channels_config, downsample=False):
        super(InceptionV2Block, self).__init__()
        if downsample:
            assert len(channels_config) == 4 
            "downsample block must have 4 channels config"
            self.add_module("b1", nn.Sequential(
                ConvBn(in_channels, channels_config[0], 1, 1),
                ConvBn(channels_config[0], channels_config[1], 3, 2, 1)
            ))
            self.add_module("b2", nn.Sequential(
                ConvBn(in_channels, channels_config[2], 1, 1),
                ConvBn(channels_config[2], channels_config[3], 3, 1, 1),
                ConvBn(channels_config[3], channels_config[3], 3, 2, 1)
            ))
            self.add_module("b3", nn.MaxPool2d(3, 2, 1))
            
        else:
            assert len(channels_config) == 6 
            "normal block must have 6 channels config"
            self.add_module("b1", ConvBn(in_channels, channels_config[0], 1, 1))
            self.add_module("b2", nn.Sequential(
                ConvBn(in_channels, channels_config[1], 1, 1),
                ConvBn(channels_config[1], channels_config[2], 3, 1, 1)
            ))
            self.add_module("b3", nn.Sequential(
                ConvBn(in_channels, channels_config[3], 1, 1),
                ConvBn(channels_config[3], channels_config[4], 3, 1, 1),
                ConvBn(channels_config[4], channels_config[4], 3, 1, 1)
            ))
            self.add_module("b4", nn.Sequential(
                nn.MaxPool2d(3, 1, 1),
                ConvBn(in_channels, channels_config[5], 1, 1)
            ))
    
    def forward(self, x):
        return torch.cat([module(x) for module in self.children()], 1)


class InceptionV2(nn.Module):
    
    def __init__(self, num_classs=1000):
        super(InceptionV2, self).__init__()
        self.add_module("stem", nn.Sequential(
            ConvBn(3, 64, 7, 2, 3),
            nn.MaxPool2d(3, 2, 1),
            ConvBn(64, 64, 1, 1),
            ConvBn(64, 192, 3, 1, 1),
            nn.MaxPool2d(3, 2, 1)
        ))
        self.add_module("inception_3a", InceptionV2Block(192, [64, 96, 128, 16, 32, 32]))
        self.add_module("inception_3b", InceptionV2Block(256, [128, 128, 192, 32, 96, 64]))
        self.add_module("inception_3c", InceptionV2Block(480, [192, 96, 208, 16, 48, 64], downsample=True))    