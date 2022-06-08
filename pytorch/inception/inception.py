import torch
import torch.nn as nn
from torchsummary import summary


class InceptionBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels_config):
        super(InceptionBlock, self).__init__()
        assert len(out_channels_config) == 6
        "out_channels_config must be 5-tuple"
        self.branch1x1 = nn.Conv2d(in_channels, out_channels_config[0], kernel_size=1)
        
        self.branch3x3_1 = nn.Conv2d(in_channels, out_channels_config[1], 1)
        self.branch3x3_2 = nn.Conv2d(out_channels_config[1], out_channels_config[2], 3, padding=1)
        
        self.branch5x5_1 = nn.Conv2d(in_channels, out_channels_config[3], 1)
        self.branch5x5_2 = nn.Conv2d(out_channels_config[3], out_channels_config[4], 5, padding=2)
        
        self.branchpool_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branchpool_2 = nn.Conv2d(in_channels, out_channels_config[5], kernel_size=1)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch1x1 = self.relu(branch1x1)
        
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.relu(branch3x3)
        
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch5x5 = self.relu(branch5x5)
        
        branchpool = self.branchpool_1(x)
        branchpool = self.branchpool_2(branchpool)
        branchpool = self.relu(branchpool)
        
        output = torch.cat([branch1x1, branch3x3, branch5x5, branchpool], 1)
        return output
        


class Inception(nn.Module):
    
    def __init__(self, num_class=1000):
        super(Inception, self).__init__()
        self.add_module("stem", nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
            nn.Conv2d(64, 192, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1)
        ))
        self.add_module("inception_3a", InceptionBlock(192, (64, 96, 128, 16, 32, 32)))
        self.add_module("inception_3b", InceptionBlock(256, (128, 128, 192, 32, 96, 64)))
        self.add_module('pool1', nn.MaxPool2d(3, 2, padding=1))
        
        self.add_module("inception_4a", InceptionBlock(480, (192, 96, 208, 16, 48, 64)))
        self.add_module("inception_4b", InceptionBlock(512, (160, 112, 224, 24, 64, 64)))
        self.add_module("inception_4c", InceptionBlock(512, (128, 128, 256, 24, 64, 64)))
        self.add_module("inception_4d", InceptionBlock(512, (112, 144, 288, 32, 64, 64)))
        self.add_module("inception_4e", InceptionBlock(528, (256, 160, 320, 32, 128, 128)))
        self.add_module("pool2", nn.MaxPool2d(3, 2, padding=1))
        
        self.add_module("inception_5a", InceptionBlock(832, (256, 160, 320, 32, 128, 128)))
        self.add_module("inception_5b", InceptionBlock(832, (384, 192, 384, 48, 128, 128)))
        self.add_module("classifier", nn.Sequential(
            nn.AvgPool2d(7, stride=1),
            nn.Flatten(),
            nn.Linear(1024, num_class),
            nn.Softmax()
        ))
        
    def forward(self, x):
        for name, module in self._modules.items():
            x = module(x)
        return x
    
if __name__ == "__main__":
    model = Inception()
    summary(model, (3, 224, 224))