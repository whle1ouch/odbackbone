from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary

class DenseLayer(nn.Module):
    
    def __init__(self, inplane, growth_rate, dropout_rate=None):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplane)
        self.conv1 = nn.Conv2d(inplane, 4 * growth_rate, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, 3, bias=False, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = dropout_rate
        
    def forward(self, x):
        y = self.bn1(x)
        y = self.relu(y)
        y = self.conv1(y)
        
        y = self.bn2(y)
        y = self.relu(y)
        y = self.conv2(y)
        if self.dropout and self.dropout > 0:
            y = F.dropout(y, p=self.dropout, training=self.training)
        y = torch.cat([x, y], 1)
        return y
    
class Transition(nn.Module):
    
    def __init__(self, inplane, outplane):
        super(Transition, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(inplane))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(inplane, outplane, 1, bias=False))
        self.add_module('pool', nn.AvgPool2d(2, 2))
        
    def forward(self, x):
        return self.pool(self.conv(self.relu(self.bn(x))))

class DenseBlock(nn.Module):

    def __init__(self, inplane, dense_num, growth_rate, dropout_rate=None):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(dense_num):
            self.layers.append(DenseLayer(inplane + i * growth_rate, growth_rate, dropout_rate))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
            
        
        
class DenseNet(nn.Module):
    def __init__(self, init_plane=64, growth_rate=32, block_config=(6, 12, 24, 16), compression=0.5, dropout_rate=None):
        super(DenseNet, self).__init__()
        self.add_module("conv0", nn.Conv2d(3, init_plane, 7, 2, 3, bias=False))
        self.add_module("bn0", nn.BatchNorm2d(init_plane))
        self.add_module("relu0", nn.ReLU(True))
        self.add_module("pool0", nn.MaxPool2d(3, 2, 1))
        inplane = init_plane
        for i in range(len(block_config)):
            self.add_module(f"denseblock{i+1}", DenseBlock(inplane, block_config[i], growth_rate, dropout_rate))
            inplane += growth_rate * block_config[i]
            if i != len(block_config) - 1:
                outplane = int(inplane * compression)
                self.add_module(f"transition{i+1}", Transition(inplane, outplane))
                inplane = outplane
        self.add_module("bn1", nn.BatchNorm2d(inplane))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("avgpool", nn.AdaptiveAvgPool2d((1, 1)))
        self.add_module('flatten', nn.Flatten(1, -1))
        self.add_module("fc", nn.Linear(inplane, 1000))
        self.add_module("softmax", nn.Softmax(dim=1))
         
        
    
    def forward(self, x):
        for name, module in self.named_children():
            x = module(x)
            print(name, x.shape)
        return x
    
def Densenet121():
    return DenseNet(block_config=(6, 12, 24, 16))


if __name__ == "__main__":
    model = Densenet121()
    torchsummary.summary(model, (3, 224, 224))
    model.eval()