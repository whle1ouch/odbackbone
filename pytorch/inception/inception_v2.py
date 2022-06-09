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
        self.add_module("inception_3a", InceptionV2Block(192, [64, 64, 64, 64, 96, 32]))
        self.add_module("inception_3b", InceptionV2Block(256, [64, 64, 64, 64, 96, 32]))
        self.add_module("inception_3c", InceptionV2Block(256, [128, 160, 64, 96], downsample=True))    
        
        self.add_module("inception_4a", InceptionV2Block(512, [224, 64, 96, 96, 128, 128]))
        self.add_module("inception_4b", InceptionV2Block(576, [192, 96, 128, 96, 128, 128]))
        self.add_module("inception_4c", InceptionV2Block(576, [160, 128, 160, 128, 160, 96]))
        self.add_module("inception_4d", InceptionV2Block(576, [96, 128, 192, 160, 192, 96]))
        self.add_module("inception_4e", InceptionV2Block(576, [128, 192, 192, 256], downsample=True))
        
        self.add_module("inception_5a", InceptionV2Block(1024, [352, 192, 320, 160, 224, 128]))
        self.add_module("inception_5b", InceptionV2Block(1024, [352, 192, 320, 192, 224, 128]))
        
        self.add_module("classifier", nn.Sequential(
            nn.AvgPool2d(7, 1),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(1024, num_classs),
            nn.Softmax()
        ))
        
    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x
    
    
if __name__ == "__main__":
    model = InceptionV2()
    summary(model, (3, 224, 224))
    